#ifndef DCPOMATIC_SLANG_J2K_ENCODER_H
#define DCPOMATIC_SLANG_J2K_ENCODER_H

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>

struct SlangJ2KEncoderImpl;

class SlangJ2KEncoder
{
public:
    SlangJ2KEncoder();
    ~SlangJ2KEncoder();
    bool is_initialized() const { return _initialized; }

    std::vector<uint8_t> encode(
        const int32_t* const xyz_planes[3],
        int width, int height,
        int64_t bit_rate, int fps, bool is_3d, bool is_4k);

private:
    std::unique_ptr<SlangJ2KEncoderImpl> _impl;
    bool _initialized = false;
    std::mutex _mutex;
};

#endif
