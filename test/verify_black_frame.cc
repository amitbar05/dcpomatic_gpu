/*
    Reproduce the "invalid JPEG2000 codestream" error from DCP export:
    encode an all-black (or near-black) 2K frame and run dcp::verify_j2k on it.
    Previously such low-content frames ended up padded with zeros to 16384 bytes
    (our minimum), and the zeros after EOC caused the verifier to throw.
*/

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>

#include "lib/cuda_j2k_encoder.h"

#include <dcp/array_data.h>
#include <dcp/verify.h>
#include <dcp/verify_j2k.h>

static void
build_identity_colour_params(GpuColourParams& p)
{
    for (int i = 0; i < 4096; ++i) {
        p.lut_in[i]  = i / 4095.0f;
        p.lut_out[i] = static_cast<uint16_t>(i);
    }
    p.matrix[0] = 0.4124f; p.matrix[1] = 0.3576f; p.matrix[2] = 0.1805f;
    p.matrix[3] = 0.2126f; p.matrix[4] = 0.7152f; p.matrix[5] = 0.0722f;
    p.matrix[6] = 0.0193f; p.matrix[7] = 0.1192f; p.matrix[8] = 0.9505f;
    p.valid = true;
}

static int
run_case(CudaJ2KEncoder& enc, const std::vector<uint16_t>& rgb,
         int W, int H, const char* label, bool fast)
{
    /* Warm up then encode */
    (void) enc.encode_ebcot(rgb.data(), W, H, W * 3, 150'000'000, 24, false, false, fast);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W * 3, 150'000'000, 24, false, false, fast);

    printf("\n=== %s (%s) ===\n", label, fast ? "fast" : "correct");
    printf("Codestream size: %zu bytes\n", cs.size());

    auto data = std::make_shared<dcp::ArrayData>(cs.data(), static_cast<int>(cs.size()));
    std::vector<dcp::VerificationNote> notes;
    dcp::verify_j2k(data, 0, 0, 24, notes);
    printf("verify_j2k notes: %zu\n", notes.size());
    for (auto const& n : notes) {
        printf("  [code=%d", int(n.code()));
        if (n.note()) printf(" note=\"%s\"", n.note()->c_str());
        printf("]\n");
    }
    return notes.empty() ? 0 : 1;
}

int main()
{
    const int W = 2048, H = 1080;
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr, "CUDA init failed\n"); return 2; }
    GpuColourParams p; build_identity_colour_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> black(static_cast<size_t>(W) * H * 3, 0);
    std::vector<uint16_t> near_black(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < near_black.size(); ++i) near_black[i] = (i % 7) == 0 ? 16 : 0;

    int fails = 0;
    fails += run_case(enc, black,      W, H, "all-black",  false);
    fails += run_case(enc, black,      W, H, "all-black",  true);
    fails += run_case(enc, near_black, W, H, "near-black", false);
    fails += run_case(enc, near_black, W, H, "near-black", true);
    printf("\n%s\n", fails == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    return fails;
}
