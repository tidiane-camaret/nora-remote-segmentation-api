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
except:
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
    """Generic cache for numpy arrays (images, ROIs, etc.)"""

    def __init__(self, max_size_bytes: int, cache_name: str = "Array"):
        self._cache = OrderedDict()
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache_name = cache_name

    def get(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: np.ndarray):
        logger = logging.getLogger(__name__)
        new_array_size = value.nbytes
        if new_array_size > self.max_size_bytes:
            raise ValueError(
                f"Array size {new_array_size} exceeds max cache size {self.max_size_bytes}"
            )

        if key in self._cache:
            old_size = self._cache[key].nbytes
            self.current_size_bytes -= old_size

        while self.current_size_bytes + new_array_size > self.max_size_bytes:
            evicted_key, evicted_value = self._cache.popitem(last=False)
            self.current_size_bytes -= evicted_value.nbytes
            logger.info(f"[CACHE {self.cache_name}] Evicted: {evicted_key}")

        self._cache[key] = value
        self.current_size_bytes += new_array_size
        logger.info(
            f"[CACHE {self.cache_name}] Stored: {key} | Cache size: {self.current_size_bytes / (1024**2):.2f} MB"
        )

    def __contains__(self, key: str) -> bool:
        return key in self._cache


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