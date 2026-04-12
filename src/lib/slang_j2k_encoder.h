#ifndef DCPOMATIC_SLANG_J2K_ENCODER_H
#define DCPOMATIC_SLANG_J2K_ENCODER_H

#include "cuda_j2k_encoder.h"   /* GpuColourParams */
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>

struct SlangJ2KEncoderImpl;

/** Slang-variant GPU JPEG2000 encoder.
 *
 *  V17 adds GPU colour conversion (same as CUDA V18):
 *  - encode_from_rgb48(): accepts RGB48LE + LUT+matrix, runs colour conversion on GPU
 *  - encode():            XYZ int32 input path (fallback if colour params not set)
 *
 *  V17 also applies CDF 9/7 analysis normalization so LL subbands do not saturate. */
class SlangJ2KEncoder
{
public:
    SlangJ2KEncoder();
    ~SlangJ2KEncoder();

    SlangJ2KEncoder(SlangJ2KEncoder const&) = delete;
    SlangJ2KEncoder& operator=(SlangJ2KEncoder const&) = delete;

    bool is_initialized()     const { return _initialized; }
    bool has_colour_params()  const { return _colour_params_valid; }

    /** Encode 3-component 12-bit XYZ image to JPEG2000 codestream (V16 path). */
    std::vector<uint8_t> encode(
        const int32_t* const xyz_planes[3],
        int width, int height,
        int64_t bit_rate, int fps, bool is_3d, bool is_4k);

    /** V17: Encode from RGB48LE with GPU colour conversion.
     *  Requires set_colour_params() called first. */
    std::vector<uint8_t> encode_from_rgb48(
        const uint16_t* rgb16,
        int width, int height,
        int rgb_stride_pixels,
        int64_t bit_rate, int fps, bool is_3d, bool is_4k);

    /** Upload colour LUT + matrix to GPU. Call once per film or per colour change. */
    void set_colour_params(GpuColourParams const& params);

    /** V17r: Flush the pipeline — collect the last in-flight frame's codestream.
     *  Must be called after the final encode_from_rgb48() to avoid losing the last frame. */
    std::vector<uint8_t> flush();

private:
    std::unique_ptr<SlangJ2KEncoderImpl> _impl;
    bool _initialized          = false;
    bool _colour_params_valid  = false;
    std::mutex _mutex;
};

std::shared_ptr<SlangJ2KEncoder> slang_j2k_encoder_instance();

#endif
