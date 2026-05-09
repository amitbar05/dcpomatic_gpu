/* Pre-include stub: defines the include guard and provides forward declarations
 * so that when gpu_ebcot_t2.h does #include "gpu_ebcot.h", the actual CUDA-dependent
 * header is skipped. */
#ifndef GPU_EBCOT_H
#define GPU_EBCOT_H

#include <cstdint>
#include <cstring>

static constexpr int CB_DIM      = 32;
static constexpr int CB_PIXELS   = CB_DIM * CB_DIM;
static constexpr int MAX_BPLANES = 16;
static constexpr int MAX_PASSES  = MAX_BPLANES * 3;
static constexpr int CB_BUF_SIZE = 16384;

static constexpr int SUBBAND_LL = 0;
static constexpr int SUBBAND_HL = 1;
static constexpr int SUBBAND_LH = 2;
static constexpr int SUBBAND_HH = 3;

struct CodeBlockInfo {
    int16_t  x0, y0;
    int16_t  width, height;
    uint8_t  subband_type;
    uint8_t  level;
    float    quant_step;
};

/* Declare a minimal tag for the T1 kernel — won't be called in tests. */
struct DWT_T { }; // placeholder

#endif
