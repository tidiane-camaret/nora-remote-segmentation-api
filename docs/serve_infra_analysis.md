# Segmentation Client-Server Logic Flaws Analysis

**Date:** October 23, 2025  
**Analyzed Components:**
- `/htdocs/KTools/KSegmentation.js` (Client)
- `/src/python/segmentation_server/main.py` (Server)

---

## Critical Issues

### 1. Hash Collision Vulnerability ⚠️ CRITICAL

**Location:** `KSegmentation.js` lines 42-47

```javascript
function simpleHash(data) {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
        hash = ((hash << 5) - hash + data[i]) | 0;
    }
    return hash.toString(16);
}
```

**Problem:**  
This simple hash function has a very high collision probability for large medical imaging datasets. Two different images could easily produce the same hash.

**Consequences:**
- Wrong cached image used for segmentation
- Segmentation results applied to incorrect patient data
- Medical data corruption in ROI
- Potential patient safety issues

**Severity:** CRITICAL - could lead to incorrect medical segmentations

**Recommended Fix:**
```javascript
async function cryptoHash(data) {
    // Use SHA-256 for proper cryptographic hashing
    const hashBuffer = await crypto.subtle.digest('SHA-256', data.buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}
```

---

### 2. Race Condition in Global State Management ⚠️ CRITICAL

**Location:** `main.py` lines 54-57

```python
PROMPT_MANAGER = PromptManager()
CURRENT_IMAGE_HASH = None  # Global state
```

**Problem:**  
The server uses global state without any locking mechanism. In a concurrent environment with multiple clients, requests can interleave.

**Example Failure Scenario:**
1. Client A uploads image X, sets `CURRENT_IMAGE_HASH = X`
2. Client B uploads image Y, sets `CURRENT_IMAGE_HASH = Y` (interleaving)
3. Client A sends ROI interaction expecting image X, but gets segmentation computed on image Y
4. Wrong segmentation returned to Client A

**Consequences:**
- Multi-client usage broken
- Segmentation computed on wrong image
- Non-deterministic behavior
- Race conditions in cache operations

**Severity:** CRITICAL - breaks multi-user deployments

**Recommended Fix:**
```python
from contextvars import ContextVar
import asyncio

# Per-request context instead of global state
_request_context: ContextVar[dict] = ContextVar('request_context', default={})

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.lock = asyncio.Lock()
    
    async def get_or_create_session(self, session_id: str):
        async with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'prompt_manager': PromptManager(),
                    'current_image_hash': None
                }
            return self.sessions[session_id]
```

---

### 3. Inconsistent Cache Check Logic ⚠️ HIGH

**Location:** `KSegmentation.js` lines 727-746

```javascript
// Check if ROI exists on server and upload if not
let roiExistsOnServer = await checkRoiOnServer(roiHash);
if (!roiExistsOnServer) {
    console.log("ROI not found on server, uploading for caching...");
    try {
        await cacheRoiOnServer(roi, uploadConfig, roiHash);
        roiExistsOnServer = true; // Assume success
    } catch (err) {
        // ...
    }
}
// ... later ...
// Assumes ROI is cached
refinedRoi = await refineSegmentationWithRoi(roi, imageHash, uploadConfig, roiHash, true);
```

**Problem:**  
Time-of-check to time-of-use (TOCTOU) vulnerability. Between checking and using the cache:
- Another process could evict the cache entry (LRU eviction)
- Server could restart
- Cache could fill up and evict the ROI
- Network errors could prevent successful caching

**Consequences:**
- HTTP 400 errors during segmentation
- Client assumes cached data exists but server doesn't have it
- Poor error recovery

**Severity:** HIGH - causes request failures

**Recommended Fix:**
- Server should accept ROI data even when cached (idempotent operations)
- Add cache version/generation numbers
- Implement atomic check-and-lock operations
- Return cache status in response headers

---

### 4. Missing Image-ROI Association Validation ⚠️ HIGH

**Location:** `main.py` lines 157-180

```python
@app.post("/add_roi_interaction")
async def add_roi_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    image_hash: str = Form(...),
    roi_hash: str = Form(...),
    file: UploadFile = File(None)
):
    pm = await ensure_active_image(image_hash)
    
    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    # Uses ROI without checking if it belongs to this image!
```

**Problem:**  
The server never validates that an ROI belongs to a specific image:
- No shape validation between image and ROI
- Different coordinate systems not checked
- Client could accidentally send ROI from image A with hash for image B

**Consequences:**
- Dimension mismatches causing crashes
- Incorrect segmentations due to misaligned data
- Array indexing errors
- Model prediction failures

**Severity:** HIGH - could cause crashes or incorrect segmentations

**Recommended Fix:**
```python
# Track image-ROI associations
IMAGE_ROI_ASSOCIATIONS = {}  # {roi_hash: (image_hash, shape)}

@app.post("/add_roi_interaction")
async def add_roi_interaction(...):
    pm = await ensure_active_image(image_hash)
    cached_roi = ROI_CACHE.get(roi_hash)
    
    # Validate association
    if roi_hash in IMAGE_ROI_ASSOCIATIONS:
        assoc_image_hash, assoc_shape = IMAGE_ROI_ASSOCIATIONS[roi_hash]
        if assoc_image_hash != image_hash:
            raise HTTPException(400, f"ROI {roi_hash} belongs to image {assoc_image_hash}, not {image_hash}")
        if assoc_shape != tuple(json.loads(shape)):
            raise HTTPException(400, "ROI shape mismatch")
```

---

## High Priority Issues

### 5. Dimension Order Confusion ⚠️ MEDIUM-HIGH

**Location:** Multiple locations in `KSegmentation.js`

```javascript
// Line 662: Client dimension ordering
const numpyarrayDims = [
    nii.sizes[2], nii.sizes[1], nii.sizes[0]  // [z, y, x] for numpy
];

// Line 398: Bbox coordinate transformation
outer_point_one: [
    Math.round(bboxParams.min[2]),  // Swaps to z,y,x
    Math.round(bboxParams.min[1]),
    Math.round(bboxParams.min[0])
]

// Line 324: Scribble coordinate transformation
scribble_coords: scribbleParams.coords.map(c => 
    [Math.round(c[2]), Math.round(c[1]), Math.round(c[0])])
```

**Problem:**  
Dimension ordering is swapped multiple times throughout the code:
- Client works internally with `[x, y, z]`
- Numpy/Server expects `[z, y, x]`
- Multiple transformations create confusion
- Easy to make mistakes when adding new features

**Consequences:**
- Misaligned segmentations
- Coordinates transformed incorrectly
- Hard to debug spatial issues
- Maintenance nightmare

**Severity:** MEDIUM-HIGH - spatial misalignment

**Recommended Fix:**
- Document coordinate conventions clearly in code comments
- Create utility functions for coordinate transformation
- Add validation tests for coordinate systems
- Use named dimensions instead of array indices

---

### 6. Coordinate Transformation Applied Incorrectly ⚠️ MEDIUM-HIGH

**Location:** `KSegmentation.js` lines 758-776

```javascript
if (mode === 'bbox') {
    // Transform bounding box coordinates to full image voxel space
    const worldToVoxel = math.inv(nii.edges);
    const center_vox = math.multiply(worldToVoxel, bboxParams.coords)._data;
    
    // Calculate half sizes in voxel space
    const halfSize = [
        bboxParams.size[0] / (2 * nii.voxSize[0]),
        bboxParams.size[1] / (2 * nii.voxSize[1]),
        bboxParams.size[2] / (2 * nii.voxSize[2])
    ];
```

**Problem:**  
The coordinate transformation logic assumes:
- `bboxParams.coords` are in world space
- `nii.edges` transform from voxel to world
- But the marker system might already provide voxel coordinates
- No validation that the transformation is correct

**Consequences:**
- Bbox coordinates might be in wrong space
- Segmentation applied to wrong region
- Size calculations might be incorrect

**Severity:** MEDIUM-HIGH - spatial errors

**Recommended Fix:**
- Document which coordinate system each API expects
- Add assertions to verify coordinate ranges
- Implement coordinate system validation
- Add unit tests for coordinate transformations

---

## Medium Priority Issues

### 7. No Server-Side Shape Validation ⚠️ MEDIUM

**Location:** `main.py` interaction endpoints

```python
@app.post("/add_bbox_interaction")
async def add_bbox_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    params: str = Form(...),
    ...
):
    bbox_params = BBoxParams(**json.loads(params))
    # Never validates that bbox coordinates are within image bounds!
```

**Problem:**  
Server accepts coordinates without validating they're within image bounds.

**Consequences:**
- Array index out of bounds errors
- Segmentation failures
- Model crashes with invalid inputs
- Poor error messages

**Severity:** MEDIUM - reduces robustness

**Recommended Fix:**
```python
def validate_bbox_in_bounds(bbox_params, image_shape):
    for i, coord in enumerate(bbox_params.outer_point_one):
        if not (0 <= coord < image_shape[i]):
            raise HTTPException(400, f"Bbox point 1 coordinate {i} out of bounds: {coord}")
    for i, coord in enumerate(bbox_params.outer_point_two):
        if not (0 <= coord < image_shape[i]):
            raise HTTPException(400, f"Bbox point 2 coordinate {i} out of bounds: {coord}")
```

---

### 8. Cache Eviction Without Notification ⚠️ MEDIUM

**Location:** `main.py` lines 36-45

```python
while self.current_size_bytes + new_array_size > self.max_size_bytes:
    # FIFO eviction: remove the oldest item
    oldest_key, oldest_value = self._cache.popitem(last=False)
    self.current_size_bytes -= oldest_value.nbytes
    print(f"{self.cache_name} cache limit exceeded. Evicted {oldest_key}...")
```

**Problem:**  
Client caches the fact that an item is on the server, but server can evict without notification.

**Consequences:**
- Client sends metadata-only requests expecting cached data
- Server returns 400 error "not in cache"
- Poor user experience
- No automatic retry or recovery

**Severity:** MEDIUM - causes request failures

**Recommended Fix:**
- Add cache generation/version numbers
- Return "cache-miss" response codes that client can handle
- Implement cache warming/pinning for active sessions
- Add cache status headers to responses

---

### 9. Unused Return Value / API Mismatch ⚠️ LOW

**Location:** `KSegmentation.js` lines 55-58

```javascript
async function processImageResponse(response) {
    const result = await response.json();
    return result.image_id;  // Returns image_id
}
```

**Location:** `main.py` line 126

```python
return {"status": "ok"}  # No image_id field!
```

**Problem:**  
Client expects `image_id` field but server doesn't provide it. The return value is unused anyway.

**Consequences:**
- Dead code
- API confusion
- Potential future bugs if code is reused

**Severity:** LOW - dead code, no functional impact

**Recommended Fix:**
- Remove unused function or fix API contract
- Add TypeScript interfaces to define API contracts
- Use API schema validation (OpenAPI/Swagger)

---

### 10. Missing Error Recovery / No Retry Logic ⚠️ MEDIUM

**Location:** `KSegmentation.js` lines 712-724

```javascript
try {
    await cacheImageOnServer(nii, imageHash, uploadConfig);
    imageExistsOnServer = true; // Assumes success
} catch (err) {
    alertify.error(err.message || "Failed to upload image.");
    if (progress_report) {
        progress_report(-1, "Upload failed");
    }
    return; // Aborts entire operation
}
```

**Problem:**  
Any transient network error causes complete operation abortion:
- No retry logic for transient failures
- No exponential backoff
- No partial recovery
- User must manually retry entire workflow

**Consequences:**
- Poor user experience on flaky networks
- Lost work from partial operations
- No resilience to transient failures

**Severity:** MEDIUM - poor user experience

**Recommended Fix:**
```javascript
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (err) {
            if (attempt === maxRetries - 1) throw err;
            const delay = baseDelay * Math.pow(2, attempt);
            console.log(`Retry attempt ${attempt + 1} after ${delay}ms`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}
```

---

## Additional Concerns

### 11. Lack of Request Timeout Handling

**Issue:** Most fetch requests don't have timeouts, which could cause indefinite hangs.

**Fix:** Add consistent timeout handling:
```javascript
const response = await fetch(url, {
    method: 'POST',
    body: formData,
    signal: AbortSignal.timeout(30000) // 30 second timeout
});
```

---

### 12. No Server-Side Request Size Limits

**Issue:** Server accepts arbitrarily large uploads without validation.

**Fix:** Add FastAPI request size limits:
```python
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_request_size=500 * 1024 * 1024  # 500 MB limit
)
```

---

### 13. Missing CORS Configuration in Production

**Location:** `main.py` line 61

```python
# TODO : restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Security issue!
    ...
)
```

**Issue:** CORS allows all origins in production.

**Fix:** Restrict origins based on deployment environment.

---

## Priority Summary

| Priority | Issue | Impact | Fix Complexity |
|----------|-------|--------|----------------|
| **CRITICAL** | Hash collision vulnerability | Data corruption, wrong segmentations | Medium |
| **CRITICAL** | Race conditions in global state | Multi-client failures | High |
| **HIGH** | Cache TOCTOU vulnerability | Request failures | Medium |
| **HIGH** | No image-ROI validation | Crashes, wrong results | Low |
| **MEDIUM-HIGH** | Dimension order confusion | Spatial misalignment | Medium |
| **MEDIUM-HIGH** | Coordinate transformation errors | Wrong regions segmented | Medium |
| **MEDIUM** | No shape validation | Crashes on invalid input | Low |
| **MEDIUM** | Cache eviction issues | Request failures | Medium |
| **MEDIUM** | No retry logic | Poor UX on network issues | Low |
| **LOW** | API mismatch | Dead code, confusion | Low |

---

## Recommended Implementation Order

1. **Fix hash collisions** - Switch to SHA-256 immediately
2. **Implement session-based architecture** - Resolve race conditions
3. **Add image-ROI association tracking** - Prevent mismatched data
4. **Implement proper error handling** - Add retries and recovery
5. **Add comprehensive validation** - Shapes, coordinates, bounds
6. **Document coordinate systems** - Prevent transformation errors
7. **Add cache management improvements** - Handle evictions gracefully
8. **Clean up dead code** - Remove unused functions
9. **Add security hardening** - CORS, rate limiting, request size limits
10. **Add comprehensive testing** - Unit tests for all critical paths

---

## Testing Recommendations

### Unit Tests Needed
- Hash collision detection tests
- Coordinate transformation validation
- Cache eviction scenarios
- Error recovery paths
- Dimension order conversions

### Integration Tests Needed
- Multi-client concurrent requests
- Cache filling and eviction
- Network failure recovery
- Timeout handling
- Image-ROI validation across API boundary

### Load Tests Needed
- Concurrent user behavior
- Cache performance under load
- Memory usage patterns
- Global state race conditions

---

**Document End**
