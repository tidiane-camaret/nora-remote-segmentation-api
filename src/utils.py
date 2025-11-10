import logging
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from huggingface_hub import snapshot_download

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# Shared constants for model and download paths
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = ".nninteractive_weights"

def download_model_weights():
    """
    Downloads model weights from Hugging Face Hub if they are not already present.
    """
    print(f"Checking for model weights: {MODEL_NAME}")

    # The download directory is relative to the project root
    model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)

    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"Model weights found in '{model_path}'. Skipping download.")
        return

    print(f"Downloading model from '{REPO_ID}' to '{DOWNLOAD_DIR}'...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR,
            local_dir_use_symlinks=False,
        )
        print("Model download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        exit(1)

if __name__ == "__main__":
    # Allows the script to be run directly to download weights.
    download_model_weights()


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_to_file: bool = False, log_level: str = "INFO"):
    """
    Configure logging for the application.

    Args:
        log_to_file: If True, logs will be written to a file. If False, only console output (default).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    handlers = []

    # Always add console handler
    handlers.append(logging.StreamHandler())

    # Optionally add file handler
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_filename = (
            log_dir / f"segmentation_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        handlers.append(logging.FileHandler(log_filename, encoding="utf-8"))

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    logger = logging.getLogger(__name__)
    if log_to_file:
        logger.info(f"Logging initialized. Log file: {log_filename}")
    else:
        logger.info("Logging initialized. Console only (file logging disabled)")

    if not GPU_AVAILABLE:
        logger.warning("GPU monitoring not available - pynvml failed to initialize")

    return logger


# ============================================================================
# Memory Monitoring (Slurm-aware)
# ============================================================================

def get_slurm_memory_limit():
    """
    Get memory limit allocated to the Slurm job.
    Returns memory limit in bytes, or None if not in a Slurm job.
    """
    logger = logging.getLogger(__name__)

    # First check Slurm environment variables (most reliable)
    if "SLURM_MEM_PER_NODE" in os.environ:
        # Memory in MB
        mem_mb = int(os.environ["SLURM_MEM_PER_NODE"])
        logger.debug(f"[SLURM DEBUG] Found SLURM_MEM_PER_NODE={mem_mb} MB")
        return mem_mb * 1024 * 1024

    if "SLURM_MEM_PER_CPU" in os.environ:
        mem_per_cpu_mb = int(os.environ["SLURM_MEM_PER_CPU"])
        # Try to get number of CPUs
        num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE",
                       os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        total_mb = mem_per_cpu_mb * num_cpus
        logger.debug(f"[SLURM DEBUG] Found SLURM_MEM_PER_CPU={mem_per_cpu_mb} MB x {num_cpus} CPUs = {total_mb} MB")
        return total_mb * 1024 * 1024

    # Try cgroup limits (only if no Slurm env vars)
    try:
        # Read /proc/self/cgroup to find the memory cgroup
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3:
                    controllers = parts[1]
                    cgroup_path = parts[2]

                    # cgroup v2
                    if controllers == "":
                        limit_file = f"/sys/fs/cgroup{cgroup_path}/memory.max"
                        if os.path.exists(limit_file):
                            with open(limit_file, "r") as f:
                                limit = f.read().strip()
                                if limit != "max" and limit.isdigit():
                                    limit_bytes = int(limit)
                                    if limit_bytes < (1024**5):  # Less than 1 PB
                                        logger.debug(f"[SLURM DEBUG] Found cgroup v2 limit: {limit_bytes / (1024**3):.2f} GB")
                                        return limit_bytes

                    # cgroup v1 memory controller
                    elif "memory" in controllers.split(","):
                        limit_file = f"/sys/fs/cgroup/memory{cgroup_path}/memory.limit_in_bytes"
                        if os.path.exists(limit_file):
                            with open(limit_file, "r") as f:
                                limit = f.read().strip()
                                if limit.isdigit():
                                    limit_bytes = int(limit)
                                    if limit_bytes < (1024**5):  # Less than 1 PB
                                        logger.debug(f"[SLURM DEBUG] Found cgroup v1 limit: {limit_bytes / (1024**3):.2f} GB")
                                        return limit_bytes
    except (OSError, ValueError) as e:
        logger.debug(f"[SLURM DEBUG] Error reading cgroup limits: {e}")

    return None


def get_job_cgroup_memory_usage():
    """
    Get the memory usage of the current cgroup (Slurm job).
    Returns memory usage in bytes, or None if not available.
    """
    logger = logging.getLogger(__name__)

    try:
        # Read /proc/self/cgroup to find the memory cgroup for this process
        with open("/proc/self/cgroup", "r") as f:
            cgroup_lines = f.readlines()

        logger.debug(f"[CGROUP DEBUG] /proc/self/cgroup contents:\n{''.join(cgroup_lines)}")

        memory_cgroup_path = None

        # Parse cgroup format
        for line in cgroup_lines:
            parts = line.strip().split(":")
            if len(parts) >= 3:
                # Format: hierarchy-ID:controller-list:cgroup-path
                controllers = parts[1]
                cgroup_path = parts[2]

                # cgroup v2: empty controller list, unified hierarchy
                if controllers == "":
                    # cgroup v2 - single unified hierarchy
                    memory_cgroup_path = cgroup_path
                    usage_file = f"/sys/fs/cgroup{memory_cgroup_path}/memory.current"
                    logger.debug(f"[CGROUP DEBUG] Trying cgroup v2: {usage_file}")
                    if os.path.exists(usage_file):
                        with open(usage_file, "r") as f:
                            usage = int(f.read().strip())
                            logger.debug(f"[CGROUP DEBUG] Read {usage / (1024**3):.2f} GB from {usage_file}")
                            return usage

                # cgroup v1: look for memory controller
                elif "memory" in controllers.split(","):
                    memory_cgroup_path = cgroup_path
                    usage_file = f"/sys/fs/cgroup/memory{memory_cgroup_path}/memory.usage_in_bytes"
                    logger.debug(f"[CGROUP DEBUG] Trying cgroup v1: {usage_file}")
                    if os.path.exists(usage_file):
                        with open(usage_file, "r") as f:
                            usage = int(f.read().strip())
                            logger.debug(f"[CGROUP DEBUG] Read {usage / (1024**3):.2f} GB from {usage_file}")
                            return usage

        logger.debug("[CGROUP DEBUG] No memory cgroup found")

    except (OSError, ValueError, IOError) as e:
        logger.debug(f"[CGROUP DEBUG] Error reading cgroup: {e}")

    return None


def log_memory_usage(context: str = ""):
    """Log current RAM and VRAM usage, respecting Slurm job limits if available."""
    logger = logging.getLogger(__name__)

    # Check if running in Slurm and get allocated memory
    slurm_limit_bytes = get_slurm_memory_limit()

    # Try to get memory used by the cgroup (includes all processes in the job)
    ram_used_bytes = get_job_cgroup_memory_usage()

    # Sanity check: if cgroup memory exceeds Slurm limit by a large margin, it's probably wrong
    if ram_used_bytes is not None and slurm_limit_bytes is not None:
        if ram_used_bytes > slurm_limit_bytes * 2:  # More than 2x the limit
            logger.debug(
                f"[CGROUP DEBUG] Cgroup memory ({ram_used_bytes / (1024**3):.2f} GB) "
                f"exceeds Slurm limit ({slurm_limit_bytes / (1024**3):.2f} GB) by >2x, "
                "cgroup reading is likely incorrect - falling back to process memory"
            )
            ram_used_bytes = None

    if ram_used_bytes is None:
        # Fall back to current process memory if cgroup reading fails
        current_process = psutil.Process()
        ram_used_bytes = current_process.memory_info().rss
        logger.debug(f"[MEMORY DEBUG] Using process RSS: {ram_used_bytes / (1024**3):.2f} GB")

    ram_used_gb = ram_used_bytes / (1024**3)

    if slurm_limit_bytes:
        # Running in Slurm job - show usage relative to job allocation
        ram_total_gb = slurm_limit_bytes / (1024**3)
        ram_percent = (ram_used_bytes / slurm_limit_bytes) * 100
        log_msg = f"[MEMORY {context}] RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_percent:.1f}% of job allocation)"
    else:
        # Not in Slurm - show system-wide memory
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        log_msg = f"[MEMORY {context}] RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_percent:.1f}% of system)"

    # VRAM usage (if GPU available)
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_gb = info.used / (1024**3)
            vram_total_gb = info.total / (1024**3)
            vram_percent = (info.used / info.total) * 100
            log_msg += f" | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB ({vram_percent:.1f}%)"
        except Exception as e:
            log_msg += f" | VRAM: Error - {str(e)}"
    else:
        log_msg += " | VRAM: N/A"

    logger.info(log_msg)


# ============================================================================
# Cache Management
# ============================================================================

class ArrayCache:
    """Generic cache for numpy arrays (images, ROIs, etc.) with disk persistence"""

    def __init__(self, max_size_bytes: int, cache_name: str = "Array", compress: bool = False, persist_dir: str | None = None, max_disk_size_bytes: int | None = None):
        self._cache = OrderedDict()
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache_name = cache_name
        self.compress = compress
        self.persist_dir = Path(persist_dir) if persist_dir else None
        # Disk cache can be larger than RAM cache
        self.max_disk_size_bytes = max_disk_size_bytes if max_disk_size_bytes else max_size_bytes
        self.current_disk_size_bytes = 0

        # Create persistence directory if specified
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _get_cache_file_path(self, key: str) -> Path:
        """Get the file path for a cache key"""
        # Sanitize key to make it filesystem-safe
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.persist_dir / f"{self.cache_name}_{safe_key}.npy"

    def _load_from_disk(self):
        """Load cached arrays from disk on initialization"""
        logger = logging.getLogger(__name__)
        if not self.persist_dir:
            return

        logger.info(f"[CACHE {self.cache_name}] Loading from disk: {self.persist_dir}")

        # Find all cache files for this cache
        pattern = f"{self.cache_name}_*.npy"
        # Sort by modification time (most recent first) to prioritize recent items
        cache_files = sorted(self.persist_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        loaded_count = 0
        disk_total = 0
        for cache_file in cache_files:
            try:
                # Extract key from filename
                key = cache_file.stem.replace(f"{self.cache_name}_", "", 1)

                # Get file size for disk tracking
                file_size = cache_file.stat().st_size
                disk_total += file_size

                # Load the array metadata/data
                if self.compress:
                    # Load compressed data stored as npz file
                    data = np.load(cache_file, allow_pickle=True)
                    compressed_data = data['compressed_data'].tobytes()
                    shape = tuple(data['shape'])
                    dtype = str(data['dtype_str'])
                    data.close()  # Close the npz file
                    stored_value = (compressed_data, shape, dtype)
                    item_size = len(stored_value[0])
                else:
                    # Load uncompressed array
                    stored_value = np.load(cache_file, mmap_mode=None)
                    item_size = stored_value.nbytes

                # Load into RAM if there's space (most recent items first)
                if self.current_size_bytes + item_size <= self.max_size_bytes:
                    self._cache[key] = stored_value
                    self.current_size_bytes += item_size
                    loaded_count += 1
                else:
                    logger.debug(f"[CACHE {self.cache_name}] {key} on disk only (RAM full)")

            except Exception as e:
                logger.warning(f"[CACHE {self.cache_name}] Failed to load {cache_file}: {e}")

        self.current_disk_size_bytes = disk_total
        logger.info(
            f"[CACHE {self.cache_name}] Loaded {loaded_count} items into RAM ({self.current_size_bytes / (1024**2):.2f} MB), "
            f"{len(cache_files)} items on disk ({self.current_disk_size_bytes / (1024**2):.2f} MB)"
        )

    def _save_to_disk(self, key: str, stored_value):
        """Save a cache entry to disk atomically, respecting disk size limit"""
        if not self.persist_dir:
            return

        logger = logging.getLogger(__name__)

        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        cache_file = self._get_cache_file_path(key)

        # Check if file already exists and track its size
        old_file_size = 0
        if cache_file.exists():
            old_file_size = cache_file.stat().st_size

        # Use a temp file with .tmp prefix to avoid .npy extension issues
        temp_file = cache_file.with_name(f".tmp_{cache_file.name}")

        try:
            if self.compress:
                # Save compressed data with metadata using npz format
                # Remove .npy extension for npz
                temp_file_npz = temp_file.with_suffix('')
                compressed_data, shape, dtype = stored_value
                np.savez(
                    str(temp_file_npz),  # numpy will add .npz
                    compressed_data=np.frombuffer(compressed_data, dtype=np.uint8),
                    shape=np.array(shape),
                    dtype_str=np.array(dtype)
                )
                # numpy adds .npz, so rename that to .npy
                actual_npz = Path(str(temp_file_npz) + '.npz')
                if actual_npz.exists():
                    new_file_size = actual_npz.stat().st_size

                    # Evict old files if needed to make space
                    self._evict_disk_space_if_needed(new_file_size - old_file_size, exclude_key=key)

                    actual_npz.replace(cache_file)
                    # Update disk usage
                    self.current_disk_size_bytes = self.current_disk_size_bytes - old_file_size + new_file_size
            else:
                # Save uncompressed array directly
                # Remove .npy extension since numpy will add it
                temp_file_no_ext = temp_file.with_suffix('')
                np.save(str(temp_file_no_ext), stored_value)
                # numpy adds .npy extension
                actual_temp = Path(str(temp_file_no_ext) + '.npy')
                if actual_temp.exists():
                    new_file_size = actual_temp.stat().st_size

                    # Evict old files if needed to make space
                    self._evict_disk_space_if_needed(new_file_size - old_file_size, exclude_key=key)

                    actual_temp.replace(cache_file)
                    # Update disk usage
                    self.current_disk_size_bytes = self.current_disk_size_bytes - old_file_size + new_file_size

            logger.debug(f"[CACHE {self.cache_name}] Saved {key} to disk (disk: {self.current_disk_size_bytes / (1024**2):.2f} MB)")

        except Exception as e:
            logger.warning(f"[CACHE {self.cache_name}] Failed to save {key} to disk: {e}")
            # Clean up any temp files
            for pattern in ['.tmp_*', '.tmp_*.npy', '.tmp_*.npz']:
                for temp in self.persist_dir.glob(pattern):
                    try:
                        temp.unlink()
                    except Exception:
                        pass

    def _evict_disk_space_if_needed(self, additional_size: int, exclude_key: str | None = None):
        """Evict oldest disk cache files if needed to make space for new item"""
        if not self.persist_dir:
            return

        logger = logging.getLogger(__name__)

        # Check if we need to evict
        if self.current_disk_size_bytes + additional_size <= self.max_disk_size_bytes:
            return

        # Find all cache files sorted by modification time (oldest first)
        pattern = f"{self.cache_name}_*.npy"
        cache_files = sorted(self.persist_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

        # Evict oldest files until we have enough space
        for cache_file in cache_files:
            if self.current_disk_size_bytes + additional_size <= self.max_disk_size_bytes:
                break

            # Extract key from filename
            key = cache_file.stem.replace(f"{self.cache_name}_", "", 1)

            # Don't evict the file we're currently saving
            if key == exclude_key:
                continue

            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                self.current_disk_size_bytes -= file_size
                logger.info(f"[CACHE {self.cache_name}] Evicted from disk: {key} ({file_size / (1024**2):.2f} MB)")

                # Also remove from RAM cache if present
                if key in self._cache:
                    if self.compress:
                        compressed_data, _, _ = self._cache[key]
                        ram_size = len(compressed_data)
                    else:
                        ram_size = self._cache[key].nbytes
                    self.current_size_bytes -= ram_size
                    del self._cache[key]

            except Exception as e:
                logger.warning(f"[CACHE {self.cache_name}] Failed to evict {key} from disk: {e}")

    def _remove_from_disk(self, key: str):
        """Remove a cache entry from disk and update disk usage tracking"""
        if not self.persist_dir:
            return

        cache_file = self._get_cache_file_path(key)
        logger = logging.getLogger(__name__)
        try:
            if cache_file.exists():
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                self.current_disk_size_bytes -= file_size
                logger.debug(f"[CACHE {self.cache_name}] Removed {key} from disk")
        except Exception as e:
            logger.warning(f"[CACHE {self.cache_name}] Failed to remove {key} from disk: {e}")

    def get(self, key: str) -> np.ndarray | None:
        # Check RAM cache first
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(key)

            if self.compress:
                # Decompress and reconstruct array
                compressed_data, shape, dtype = self._cache[key]
                return deserialize_array(compressed_data, shape, dtype, compressed=True, log_stats=False)
            else:
                return self._cache[key]

        # Not in RAM - check disk if persistence is enabled
        if self.persist_dir:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                logger = logging.getLogger(__name__)
                logger.debug(f"[CACHE {self.cache_name}] Loading {key} from disk")

                try:
                    # Load from disk
                    if self.compress:
                        data = np.load(cache_file, allow_pickle=True)
                        compressed_data = data['compressed_data'].tobytes()
                        shape = tuple(data['shape'])
                        dtype = str(data['dtype_str'])
                        data.close()
                        stored_value = (compressed_data, shape, dtype)
                        item_size = len(compressed_data)

                        # Try to add to RAM cache if it fits
                        if item_size <= self.max_size_bytes:
                            # Evict items if needed
                            while self.current_size_bytes + item_size > self.max_size_bytes:
                                if not self._cache:
                                    break
                                evicted_key, evicted_value = self._cache.popitem(last=False)
                                evicted_compressed_data, _, _ = evicted_value
                                evicted_size = len(evicted_compressed_data)
                                self.current_size_bytes -= evicted_size
                                logger.debug(f"[CACHE {self.cache_name}] Evicted {evicted_key} to make room")

                            self._cache[key] = stored_value
                            self.current_size_bytes += item_size
                            logger.debug(f"[CACHE {self.cache_name}] Loaded {key} into RAM")

                        # Return decompressed array
                        return deserialize_array(compressed_data, shape, dtype, compressed=True, log_stats=False)
                    else:
                        # Uncompressed array
                        array = np.load(cache_file, mmap_mode=None)
                        item_size = array.nbytes

                        # Try to add to RAM cache if it fits
                        if item_size <= self.max_size_bytes:
                            # Evict items if needed
                            while self.current_size_bytes + item_size > self.max_size_bytes:
                                if not self._cache:
                                    break
                                evicted_key, evicted_value = self._cache.popitem(last=False)
                                evicted_size = evicted_value.nbytes
                                self.current_size_bytes -= evicted_size
                                logger.debug(f"[CACHE {self.cache_name}] Evicted {evicted_key} to make room")

                            self._cache[key] = array
                            self.current_size_bytes += item_size
                            logger.debug(f"[CACHE {self.cache_name}] Loaded {key} into RAM")

                        return array

                except Exception as e:
                    logger.warning(f"[CACHE {self.cache_name}] Failed to load {key} from disk: {e}")

        return None

    def set(self, key: str, value: np.ndarray):
        logger = logging.getLogger(__name__)

        if self.compress:
            # Compress the array and store with metadata
            compressed_data = serialize_array(value, compress=True, pack_bits=False)
            new_item_size = len(compressed_data)
            stored_value = (compressed_data, value.shape, str(value.dtype))

            original_size = value.nbytes
            compression_ratio = (1 - new_item_size / original_size) * 100
            logger.debug(
                f"[CACHE {self.cache_name}] Compressed {key}: "
                f"{original_size / (1024**2):.2f} MB -> {new_item_size / (1024**2):.2f} MB "
                f"({compression_ratio:.1f}% compression)"
            )
        else:
            # Store array directly
            new_item_size = value.nbytes
            stored_value = value

        # Check against disk size if persistence is enabled, otherwise check against RAM
        size_limit = self.max_disk_size_bytes if self.persist_dir else self.max_size_bytes
        if new_item_size > size_limit:
            raise ValueError(
                f"Item size {new_item_size / (1024**2):.2f} MB exceeds max cache size {size_limit / (1024**2):.2f} MB"
            )

        # Remove old item if key exists
        if key in self._cache:
            if self.compress:
                old_compressed_data, _, _ = self._cache[key]
                old_size = len(old_compressed_data)
            else:
                old_size = self._cache[key].nbytes
            self.current_size_bytes -= old_size

        # Evict items from RAM if necessary
        while self.current_size_bytes + new_item_size > self.max_size_bytes:
            if not self._cache:
                # No items in RAM to evict
                break
            evicted_key, evicted_value = self._cache.popitem(last=False)
            if self.compress:
                evicted_compressed_data, _, _ = evicted_value
                evicted_size = len(evicted_compressed_data)
            else:
                evicted_size = evicted_value.nbytes
            self.current_size_bytes -= evicted_size
            logger.info(f"[CACHE {self.cache_name}] Evicted from RAM: {evicted_key}")
            # Note: Don't remove from disk - it stays there for later retrieval

        # Store the item in memory only if it fits in RAM
        if new_item_size <= self.max_size_bytes:
            self._cache[key] = stored_value
            self.current_size_bytes += new_item_size
            logger.info(
                f"[CACHE {self.cache_name}] Stored: {key} | "
                f"RAM: {self.current_size_bytes / (1024**2):.2f} MB"
            )
        else:
            logger.info(
                f"[CACHE {self.cache_name}] Item {key} too large for RAM "
                f"({new_item_size / (1024**2):.2f} MB), storing on disk only"
            )

        # Save to disk if persistence is enabled
        self._save_to_disk(key, stored_value)

    def __contains__(self, key: str) -> bool:
        # Check RAM cache first
        if key in self._cache:
            return True
        # Check disk if persistence is enabled
        if self.persist_dir:
            cache_file = self._get_cache_file_path(key)
            return cache_file.exists()
        return False


# ============================================================================
# Binary Compression & Array Serialization
# ============================================================================

def compress_binary(data: bytes) -> bytes:
    """Compress binary data using gzip."""
    import gzip
    return gzip.compress(data)


def decompress_binary(data: bytes, log_stats: bool = True) -> bytes:
    """
    Decompress gzip binary data.

    Args:
        data: Compressed binary data
        log_stats: If True, log compression statistics

    Returns:
        Decompressed binary data
    """
    import gzip

    logger = logging.getLogger(__name__)
    original_size = len(data)
    decompressed_data = gzip.decompress(data)

    if log_stats:
        decompressed_size = len(decompressed_data)
        compression_ratio = (1 - original_size / decompressed_size) * 100
        logger.info(
            f"Decompressed: {original_size} bytes -> {decompressed_size} bytes "
            f"({compression_ratio:.2f}% compression)"
        )

    return decompressed_data


def serialize_array(array: np.ndarray, compress: bool = False, pack_bits: bool = False) -> bytes:
    """
    Serialize a numpy array to bytes with optional compression.

    Args:
        array: Numpy array to serialize
        compress: If True, compress the output with gzip
        pack_bits: If True, pack boolean array using np.packbits (only for boolean arrays)

    Returns:
        Binary data (optionally compressed)
    """
    if pack_bits:
        # Special handling for boolean arrays (segmentation masks)
        array = array.astype(bool)
        binary_data = np.packbits(array, axis=None).tobytes()
    else:
        # General array serialization
        binary_data = array.tobytes()

    if compress:
        binary_data = compress_binary(binary_data)

    return binary_data


def deserialize_array(
    binary_data: bytes,
    shape: tuple,
    dtype: str,
    compressed: bool = False,
    log_stats: bool = True
) -> np.ndarray:
    """
    Deserialize binary data to a numpy array with optional decompression.

    Args:
        binary_data: Binary data to deserialize
        shape: Shape of the output array
        dtype: Data type of the array
        compressed: If True, decompress data first
        log_stats: If True, log decompression statistics

    Returns:
        Numpy array
    """
    if compressed:
        binary_data = decompress_binary(binary_data, log_stats=log_stats)

    return np.frombuffer(binary_data, dtype=np.dtype(dtype)).reshape(shape)