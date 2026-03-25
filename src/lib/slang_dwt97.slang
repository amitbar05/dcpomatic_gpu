// JPEG2000 DWT 9/7 Wavelet Transform - Slang Compute Shader
//
// Implements the CDF 9/7 irreversible wavelet transform used in JPEG2000
// lossy encoding. Compiles to CUDA via: slangc -target cuda
//
// This shader performs:
//   1. Horizontal DWT 9/7 lifting steps
//   2. Horizontal deinterleave (split into L/H subbands)
//   3. Vertical DWT 9/7 lifting steps
//   4. Vertical deinterleave
//   5. Quantization of wavelet coefficients

// CDF 9/7 lifting coefficients
static const float ALPHA = -1.586134342f;
static const float BETA  = -0.052980118f;
static const float GAMMA =  0.882911075f;
static const float DELTA =  0.443506852f;

// Buffers bound from host
[vk::binding(0)] RWStructuredBuffer<float> data;
[vk::binding(1)] RWStructuredBuffer<float> tmp;
[vk::binding(2)] RWStructuredBuffer<int>   inputBuf;
[vk::binding(3)] RWStructuredBuffer<int>   quantOut;

// Push constants for dimensions
struct PushConstants {
    int width;
    int height;
    int stride;
    float stepSize;
    float alpha;
    int phase;
};

[[vk::push_constant]] PushConstants pc;


// ==========================================
// Kernel: Convert int32 XYZ data to float
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void intToFloat(uint3 tid : SV_DispatchThreadID)
{
    int idx = int(tid.x);
    int n = pc.width * pc.height;
    if (idx < n) {
        data[idx] = float(inputBuf[idx]);
    }
}


// ==========================================
// Kernel: Horizontal DWT 9/7 lifting step
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void dwt97HorzStep(uint3 tid : SV_DispatchThreadID)
{
    int y = int(tid.x);
    if (y >= pc.height) return;

    int w = pc.width;
    int s = pc.stride;
    float a = pc.alpha;

    if (pc.phase == 0) {
        // Update odd samples
        for (int x = 1; x < w - 1; x += 2) {
            data[y * s + x] += a * (data[y * s + x - 1] + data[y * s + x + 1]);
        }
        if (w > 1 && (w % 2 == 0)) {
            data[y * s + w - 1] += 2.0f * a * data[y * s + w - 2];
        }
    } else {
        // Update even samples
        for (int x = 2; x < w; x += 2) {
            int xp1 = (x + 1 < w) ? x + 1 : x - 1;
            data[y * s + x] += a * (data[y * s + x - 1] + data[y * s + xp1]);
        }
        data[y * s + 0] += 2.0f * a * data[y * s + min(1, w - 1)];
    }
}


// ==========================================
// Kernel: Vertical DWT 9/7 lifting step
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void dwt97VertStep(uint3 tid : SV_DispatchThreadID)
{
    int x = int(tid.x);
    if (x >= pc.width) return;

    int h = pc.height;
    int s = pc.stride;
    float a = pc.alpha;

    if (pc.phase == 0) {
        for (int y = 1; y < h - 1; y += 2) {
            data[y * s + x] += a * (data[(y - 1) * s + x] + data[(y + 1) * s + x]);
        }
        if (h > 1 && (h % 2 == 0)) {
            data[(h - 1) * s + x] += 2.0f * a * data[(h - 2) * s + x];
        }
    } else {
        for (int y = 2; y < h; y += 2) {
            int yp1 = (y + 1 < h) ? y + 1 : y - 1;
            data[y * s + x] += a * (data[(y - 1) * s + x] + data[yp1 * s + x]);
        }
        data[0 * s + x] += 2.0f * a * data[min(1, h - 1) * s + x];
    }
}


// ==========================================
// Kernel: Horizontal deinterleave
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void deinterleaveHorz(uint3 tid : SV_DispatchThreadID)
{
    int y = int(tid.x);
    if (y >= pc.height) return;

    int w = pc.width;
    int s = pc.stride;
    int halfW = (w + 1) / 2;

    // Even samples -> low-pass (left half)
    for (int x = 0; x < w; x += 2) {
        tmp[y * s + x / 2] = data[y * s + x];
    }
    // Odd samples -> high-pass (right half)
    for (int x = 1; x < w; x += 2) {
        tmp[y * s + halfW + x / 2] = data[y * s + x];
    }
}


// ==========================================
// Kernel: Vertical deinterleave
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void deinterleaveVert(uint3 tid : SV_DispatchThreadID)
{
    int x = int(tid.x);
    if (x >= pc.width) return;

    int h = pc.height;
    int s = pc.stride;
    int halfH = (h + 1) / 2;

    for (int y = 0; y < h; y += 2) {
        tmp[(y / 2) * s + x] = data[y * s + x];
    }
    for (int y = 1; y < h; y += 2) {
        tmp[(halfH + y / 2) * s + x] = data[y * s + x];
    }
}


// ==========================================
// Kernel: Copy tmp -> data
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void copyTmpToData(uint3 tid : SV_DispatchThreadID)
{
    int idx = int(tid.x);
    int n = pc.width * pc.height;
    if (idx < n) {
        data[idx] = tmp[idx];
    }
}


// ==========================================
// Kernel: Quantize wavelet coefficients
// ==========================================
[shader("compute")]
[numthreads(256, 1, 1)]
void quantize(uint3 tid : SV_DispatchThreadID)
{
    int idx = int(tid.x);
    int n = pc.width * pc.height;
    if (idx < n) {
        float val = data[idx] / pc.stepSize;
        quantOut[idx] = int(round(val));
    }
}
