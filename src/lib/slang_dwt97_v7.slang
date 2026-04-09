// JPEG2000 DWT 9/7 Wavelet Transform - Slang Compute Shader v7
//
// Portable GPU shader: compiles to CUDA, SPIR-V, HLSL, Metal, WGSL, C++
//   slangc slang_dwt97_v7.slang -target cuda  -entry <name> -stage compute -o out.cu
//   slangc slang_dwt97_v7.slang -target spirv -entry <name> -stage compute -o out.spv
//   slangc slang_dwt97_v7.slang -target hlsl  -entry <name> -stage compute -o out.hlsl
//
// Key optimization: shared-memory horizontal DWT (all 4 lifting steps in smem)

// CDF 9/7 lifting coefficients
static const float ALPHA = -1.586134342f;
static const float BETA  = -0.052980118f;
static const float GAMMA =  0.882911075f;
static const float DELTA =  0.443506852f;

// GPU buffers
[vk::binding(0)] RWStructuredBuffer<float> g_data;
[vk::binding(1)] RWStructuredBuffer<float> g_tmp;
[vk::binding(2)] RWStructuredBuffer<int>   g_input;
[vk::binding(3)] RWStructuredBuffer<uint8_t> g_encoded;

struct Params {
    int width;
    int height;
    int stride;
    int n;           // width * height
    float stepSize;
    int targetN;     // target encoded size
};

[[vk::push_constant]] Params p;


// =====================
// int32 -> float
// =====================
[shader("compute")]
[numthreads(128, 1, 1)]
void intToFloat(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i < p.n)
        g_data[i] = float(g_input[i]);
}


// =====================
// Shared-memory horizontal DWT 9/7
// One workgroup per row. Thread 0 does the lifting in groupshared memory.
// All threads cooperatively load/store.
// =====================
groupshared float smem[2048];  // max 2K width

[shader("compute")]
[numthreads(128, 1, 1)]
void dwtHorzSmem(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID)
{
    int y = int(gid.x);
    if (y >= p.height) return;

    int tid = int(ltid.x);
    int nt = 128;
    int w = p.width;
    int s = p.stride;

    // Cooperative load row into shared memory
    for (int i = tid; i < w; i += nt)
        smem[i] = g_data[y * s + i];
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 performs all 4 lifting steps in shared memory
    if (tid == 0) {
        // Alpha on odd
        for (int x = 1; x < w - 1; x += 2)
            smem[x] += ALPHA * (smem[x-1] + smem[x+1]);
        if (w > 1 && (w % 2 == 0))
            smem[w-1] += 2.0f * ALPHA * smem[w-2];

        // Beta on even
        smem[0] += 2.0f * BETA * smem[min(1, w-1)];
        for (int x = 2; x < w; x += 2)
            smem[x] += BETA * (smem[x-1] + smem[min(x+1, w-1)]);

        // Gamma on odd
        for (int x = 1; x < w - 1; x += 2)
            smem[x] += GAMMA * (smem[x-1] + smem[x+1]);
        if (w > 1 && (w % 2 == 0))
            smem[w-1] += 2.0f * GAMMA * smem[w-2];

        // Delta on even
        smem[0] += 2.0f * DELTA * smem[min(1, w-1)];
        for (int x = 2; x < w; x += 2)
            smem[x] += DELTA * (smem[x-1] + smem[min(x+1, w-1)]);
    }
    GroupMemoryBarrierWithGroupSync();

    // Cooperative deinterleave write
    int hw = (w + 1) / 2;
    for (int x = tid * 2; x < w; x += nt * 2)
        g_tmp[y * s + x / 2] = smem[x];
    for (int x = tid * 2 + 1; x < w; x += nt * 2)
        g_tmp[y * s + hw + x / 2] = smem[x];
}


// =====================
// Fallback horizontal DWT (for rows > 2048)
// =====================
[shader("compute")]
[numthreads(128, 1, 1)]
void dwtHorz(uint3 tid : SV_DispatchThreadID)
{
    int y = int(tid.x);
    if (y >= p.height) return;

    int w = p.width;
    int s = p.stride;
    int base = y * s;

    for (int x = 1; x < w - 1; x += 2)
        g_data[base+x] += ALPHA * (g_data[base+x-1] + g_data[base+x+1]);
    if (w > 1 && (w % 2 == 0))
        g_data[base+w-1] += 2.0f * ALPHA * g_data[base+w-2];

    g_data[base] += 2.0f * BETA * g_data[base + min(1, w-1)];
    for (int x = 2; x < w; x += 2)
        g_data[base+x] += BETA * (g_data[base+x-1] + g_data[base + min(x+1, w-1)]);

    for (int x = 1; x < w - 1; x += 2)
        g_data[base+x] += GAMMA * (g_data[base+x-1] + g_data[base+x+1]);
    if (w > 1 && (w % 2 == 0))
        g_data[base+w-1] += 2.0f * GAMMA * g_data[base+w-2];

    g_data[base] += 2.0f * DELTA * g_data[base + min(1, w-1)];
    for (int x = 2; x < w; x += 2)
        g_data[base+x] += DELTA * (g_data[base+x-1] + g_data[base + min(x+1, w-1)]);

    int hw = (w + 1) / 2;
    for (int x = 0; x < w; x += 2)
        g_tmp[base + x/2] = g_data[base + x];
    for (int x = 1; x < w; x += 2)
        g_tmp[base + hw + x/2] = g_data[base + x];
}


// =====================
// Vertical DWT 9/7
// =====================
[shader("compute")]
[numthreads(128, 1, 1)]
void dwtVert(uint3 tid : SV_DispatchThreadID)
{
    int x = int(tid.x);
    if (x >= p.width) return;

    int h = p.height;
    int s = p.stride;

    // Alpha on odd rows
    for (int y = 1; y < h - 1; y += 2)
        g_data[y*s+x] += ALPHA * (g_data[(y-1)*s+x] + g_data[(y+1)*s+x]);
    if (h > 1 && (h % 2 == 0))
        g_data[(h-1)*s+x] += 2.0f * ALPHA * g_data[(h-2)*s+x];

    // Beta on even rows
    g_data[x] += 2.0f * BETA * g_data[min(1,h-1)*s+x];
    for (int y = 2; y < h; y += 2)
        g_data[y*s+x] += BETA * (g_data[(y-1)*s+x] + g_data[min(y+1,h-1)*s+x]);

    // Gamma on odd rows
    for (int y = 1; y < h - 1; y += 2)
        g_data[y*s+x] += GAMMA * (g_data[(y-1)*s+x] + g_data[(y+1)*s+x]);
    if (h > 1 && (h % 2 == 0))
        g_data[(h-1)*s+x] += 2.0f * GAMMA * g_data[(h-2)*s+x];

    // Delta on even rows
    g_data[x] += 2.0f * DELTA * g_data[min(1,h-1)*s+x];
    for (int y = 2; y < h; y += 2)
        g_data[y*s+x] += DELTA * (g_data[(y-1)*s+x] + g_data[min(y+1,h-1)*s+x]);

    // Deinterleave
    int hh = (h + 1) / 2;
    for (int y = 0; y < h; y += 2)
        g_tmp[(y/2)*s+x] = g_data[y*s+x];
    for (int y = 1; y < h; y += 2)
        g_tmp[(hh+y/2)*s+x] = g_data[y*s+x];
}


// =====================
// Quantize + encode to bytes
// =====================
[shader("compute")]
[numthreads(128, 1, 1)]
void quantEncode(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i >= p.targetN) return;

    if (i < p.n) {
        int q = int(round(g_data[i] / p.stepSize));
        uint8_t sign = (q < 0) ? uint8_t(0x80) : uint8_t(0);
        uint8_t mag = uint8_t(min(127, abs(q)));
        g_encoded[i] = sign | mag;
    } else {
        g_encoded[i] = uint8_t(0);
    }
}
