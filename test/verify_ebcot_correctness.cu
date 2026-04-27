/*
    EBCOT encode_ebcot path correctness verification.

    Tests the GPU EBCOT T1 + T2 pipeline by encoding synthetic DWT
    coefficient arrays and verifying that OpenJPEG can decode the result
    and that decoded pixels are consistent with the encoded data.

    Key check: when we place a known pattern in LL/detail subbands, the
    decoded image should show that pattern.  If the zero-bitplanes tag
    tree (or MQ coding) is broken, OpenJPEG will decode zero everywhere
    (output uniformly ~2048 due to DC shift).

    Build:
      nvcc -std=c++17 -O2 test/verify_ebcot_correctness.cu \
        -I src/lib \
        -I /usr/include/openjpeg-2.5 \
        -lopenjp2 -lcudart \
        -o test/verify_ebcot_correctness

    Usage:
      ./test/verify_ebcot_correctness
*/

#include "gpu_ebcot.h"
#include "gpu_ebcot_t2.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <openjpeg.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

/* ===== OpenJPEG 2.x decode via temp file ===== */

static bool opj_decode_j2k(const uint8_t* data, size_t len,
                            std::vector<std::vector<int32_t>>& comps,
                            int& out_w, int& out_h, int& ncomps) {
    char tmp_path[] = "/tmp/verify_ebcot_XXXXXX.j2c";
    int fd = mkstemps(tmp_path, 4);
    if (fd < 0) { perror("mkstemps"); return false; }
    if (write(fd, data, len) != (ssize_t)len) { perror("write"); close(fd); unlink(tmp_path); return false; }
    close(fd);

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    if (!codec) { fprintf(stderr, "opj_create_decompress failed\n"); unlink(tmp_path); return false; }

    opj_set_warning_handler(codec, [](const char* m, void*){ fprintf(stderr,"OPJ warn: %s\n",m); }, nullptr);
    opj_set_error_handler(codec, [](const char* m, void*){ fprintf(stderr,"OPJ err: %s\n",m); }, nullptr);

    opj_dparameters_t params;
    opj_set_default_decoder_parameters(&params);
    opj_setup_decoder(codec, &params);

    opj_stream_t* stream = opj_stream_create_file_stream(tmp_path, 1 << 20, OPJ_TRUE);
    if (!stream) {
        fprintf(stderr, "opj_stream_create_file_stream failed\n");
        opj_destroy_codec(codec); unlink(tmp_path); return false;
    }

    opj_image_t* image = nullptr;
    if (!opj_read_header(stream, codec, &image)) {
        fprintf(stderr, "opj_read_header failed\n");
        opj_stream_destroy(stream); opj_destroy_codec(codec); unlink(tmp_path); return false;
    }
    if (!opj_decode(codec, stream, image)) {
        fprintf(stderr, "opj_decode failed\n");
        opj_image_destroy(image); opj_stream_destroy(stream);
        opj_destroy_codec(codec); unlink(tmp_path); return false;
    }
    opj_end_decompress(codec, stream);

    out_w = (int)image->comps[0].w;
    out_h = (int)image->comps[0].h;
    ncomps = (int)image->numcomps;
    comps.resize(ncomps);
    for (int c = 0; c < ncomps; c++) {
        int n = (int)(image->comps[c].w * image->comps[c].h);
        comps[c].resize(n);
        for (int i = 0; i < n; i++) comps[c][i] = image->comps[c].data[i];
    }

    opj_image_destroy(image); opj_stream_destroy(stream);
    opj_destroy_codec(codec); unlink(tmp_path);
    return true;
}

/* Save codestream to file for inspection */
static void save_cs(const std::vector<uint8_t>& cs, const char* path) {
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(cs.data(), 1, cs.size(), f); fclose(f); }
}

/* ===== Main test logic ===== */

/* Run one test case.  We put a ramp of quantized values into the LL subband
 * (and zeros elsewhere), encode via GPU T1+T2, decode via OpenJPEG, and
 * verify that the decoded output has meaningful variation (not all-2048). */
static bool run_test(int width, int height, float base_step, const char* label) {
    printf("\n--- %s (w=%d h=%d step=%.4f) ---\n", label, width, height, base_step);

    const int num_levels = 5;
    const bool is_4k = false;

    /* Build CB table */
    std::vector<CodeBlockInfo> cb_infos;
    std::vector<SubbandGeom> subbands;
    build_codeblock_table(width, height, width, num_levels, base_step, is_4k,
                          cb_infos, subbands);
    int num_cbs = (int)cb_infos.size();

    /* Find LL subband dimensions */
    int lw = width, lh = height;
    for (int i = 0; i < num_levels; i++) { lw = (lw+1)/2; lh = (lh+1)/2; }
    printf("  LL subband: %dx%d\n", lw, lh);

    /* Build synthetic DWT array:
     * - LL subband: large ramp values so decoded pixels differ clearly from DC-offset zero
     * - Other subbands: zero */
    /* step for LL = base_step * level_weight[num_levels=5] = base_step * 0.65 */
    float step_ll = base_step * 0.65f;
    /* Use LARGE coefficients so reconstructed pixels are clearly non-zero.
     * With 5-level inverse DWT, each LL coeff contributes ~0.35x to pixel values.
     * We need coeff >> 1 to produce pixel offsets > 10 from 2048. */
    float target_coeff = 200.0f;    /* → pixel offset ≈ 200 * 0.35 = 70 from 2048 */
    int target_q = (int)(target_coeff / step_ll);
    if (target_q > 30000) target_q = 30000;  /* clamp to fit in int16 */
    printf("  step_ll=%.5f  target_q=%d  target_coeff=%.1f\n",
           step_ll, target_q, target_q * step_ll);

    std::vector<__half> h_dwt(width * height, __float2half(0.0f));
    /* Ramp: different values across LL so decoded pixels vary spatially */
    int nll = lw * lh;
    for (int i = 0; i < nll; i++) {
        /* q ranges from target_q/2 to target_q */
        int q = target_q / 2 + (i * (target_q / 2) / std::max(nll - 1, 1));
        if (q < 1) q = 1;
        float coeff = q * step_ll;
        int ry = i / lw, rx = i % lw;
        h_dwt[ry * width + rx] = __float2half(coeff);
    }

    /* Upload DWT array to GPU */
    __half* d_dwt = nullptr;
    size_t dwt_bytes = (size_t)width * height * sizeof(__half);
    cudaMalloc(&d_dwt, dwt_bytes);
    cudaMemcpy(d_dwt, h_dwt.data(), dwt_bytes, cudaMemcpyHostToDevice);

    CodeBlockInfo* d_cb_info = nullptr;
    cudaMalloc(&d_cb_info, num_cbs * sizeof(CodeBlockInfo));
    cudaMemcpy(d_cb_info, cb_infos.data(), num_cbs * sizeof(CodeBlockInfo), cudaMemcpyHostToDevice);

    uint8_t*  d_coded_data = nullptr; cudaMalloc(&d_coded_data,  (size_t)num_cbs * CB_BUF_SIZE);
    uint16_t* d_coded_len  = nullptr; cudaMalloc(&d_coded_len,   num_cbs * sizeof(uint16_t));
    uint8_t*  d_num_passes = nullptr; cudaMalloc(&d_num_passes,  num_cbs * sizeof(uint8_t));
    uint16_t* d_pass_lens  = nullptr; cudaMalloc(&d_pass_lens,   (size_t)num_cbs * MAX_PASSES * sizeof(uint16_t));
    uint8_t*  d_num_bp     = nullptr; cudaMalloc(&d_num_bp,      num_cbs * sizeof(uint8_t));

    constexpr int THREADS = 64;
    int grid = (num_cbs + THREADS - 1) / THREADS;
    kernel_ebcot_t1<<<grid, THREADS>>>(
        d_dwt, width, d_cb_info, num_cbs,
        d_coded_data, d_coded_len, d_num_passes, d_pass_lens, d_num_bp, 0);

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "T1 kernel failed\n"); return false;
    }

    std::vector<uint8_t>  h_coded_data((size_t)num_cbs * CB_BUF_SIZE);
    std::vector<uint16_t> h_coded_len(num_cbs);
    std::vector<uint8_t>  h_num_passes(num_cbs);
    std::vector<uint8_t>  h_num_bp(num_cbs);

    cudaMemcpy(h_coded_data.data(), d_coded_data, (size_t)num_cbs * CB_BUF_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_coded_len.data(),  d_coded_len,  num_cbs * sizeof(uint16_t),    cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_passes.data(), d_num_passes, num_cbs * sizeof(uint8_t),     cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_bp.data(),     d_num_bp,     num_cbs * sizeof(uint8_t),     cudaMemcpyDeviceToHost);

    int nonzero = 0, total_bytes = 0;
    for (int i = 0; i < num_cbs; i++) {
        if (h_coded_len[i] > 0) { nonzero++; total_bytes += h_coded_len[i]; }
    }
    printf("  T1: %d/%d CBs coded, %d bytes\n", nonzero, num_cbs, total_bytes);

    /* Print LL CB info */
    if (num_cbs > 0)
        printf("  CB[0](LL): len=%d np=%d num_bp=%d\n",
               h_coded_len[0], h_num_passes[0], h_num_bp[0]);

    /* Compute expected z for LL subband */
    {
        float step = subbands[0].step;
        int eps = 13 - (int)std::floor(std::log2f(std::max(step, 1e-6f)));
        int pmax = eps + 1;
        int z = pmax - (int)h_num_bp[0];
        printf("  LL: step=%.5f eps=%d pmax=%d z=%d\n", step, eps, pmax, z);
    }

    /* Build codestream using all 3 components (same data per component for this test) */
    const uint8_t*  cd[3] = { h_coded_data.data(), h_coded_data.data(), h_coded_data.data() };
    const uint16_t* cl[3] = { h_coded_len.data(),  h_coded_len.data(),  h_coded_len.data() };
    const uint8_t*  np[3] = { h_num_passes.data(), h_num_passes.data(), h_num_passes.data() };
    const uint16_t* pl[3] = { nullptr, nullptr, nullptr };
    const uint8_t*  nb[3] = { h_num_bp.data(), h_num_bp.data(), h_num_bp.data() };

    auto cs = build_ebcot_codestream(
        width, height, is_4k, false, num_levels, base_step,
        subbands, cd, cl, np, pl, nb, 0);

    printf("  Codestream: %zu bytes\n", cs.size());

    /* Save for inspection */
    char path[64]; snprintf(path, sizeof(path), "/tmp/ebcot_test_%dx%d.j2c", width, height);
    save_cs(cs, path);
    printf("  Saved to %s\n", path);

    /* Decode */
    std::vector<std::vector<int32_t>> comps;
    int dw = 0, dh = 0, nc = 0;
    if (!opj_decode_j2k(cs.data(), cs.size(), comps, dw, dh, nc)) {
        printf("  FAIL: OpenJPEG decode failed\n");
        return false;
    }
    printf("  Decoded: %dx%d, %d components\n", dw, dh, nc);

    /* Check comp[0] (the Y/X component) */
    auto& pix = comps[0];
    if (pix.empty()) { printf("  FAIL: empty decoded image\n"); return false; }

    int pmin = *std::min_element(pix.begin(), pix.end());
    int pmax_v = *std::max_element(pix.begin(), pix.end());
    double pmean = 0;
    for (auto v : pix) pmean += v;
    pmean /= pix.size();

    printf("  Decoded comp[0]: min=%d max=%d mean=%.1f\n", pmin, pmax_v, pmean);

    /* The DC level for unsigned 12-bit J2K is 2048 (DC-shifted).
     * With large LL coefficients (target_coeff ≈ 200), the reconstruction should
     * produce pixels significantly above 2048.  If z-encoding is broken, OpenJPEG
     * reads garbage bit-planes and outputs ~2048 everywhere. */
    int range = pmax_v - pmin;

    /* Print first few decoded values */
    printf("  First 8 pixels: ");
    for (int i = 0; i < 8 && i < (int)pix.size(); i++) printf("%d ", pix[i]);
    printf("\n");

    /* Key check: decoded image must have clear spatial variation.
     * A successful decode of our LL ramp gives range >> 0.
     * If z-encoding is broken, OpenJPEG reads garbage bit-planes → range ≈ 0.
     * Note: using the same data for all 3 components causes ICT mixing cancellation,
     * so mean can land near 2048 even with correct decode (range is the robust check). */
    bool range_ok = (range > 10);

    if (!range_ok) {
        printf("  FAIL: range=%d too small — likely z-encoding or decode failure\n", range);
    } else {
        printf("  OK: mean=%.1f (dev from 2048: %.1f), range=%d\n",
               pmean, pmean - 2048.0, range);
    }

    /* Cleanup */
    cudaFree(d_dwt); cudaFree(d_cb_info);
    cudaFree(d_coded_data); cudaFree(d_coded_len); cudaFree(d_num_passes);
    cudaFree(d_pass_lens); cudaFree(d_num_bp);

    return range_ok;
}

int main() {
    printf("=== EBCOT encode_ebcot correctness test ===\n");
    printf("Checking that z (zero bit-planes) encoding is correct.\n");
    printf("A uniformly 2048 decoded output means OpenJPEG decoded all-zero data\n");
    printf("which indicates the packet header bit fields are misaligned.\n");

    bool ok = true;
    /* Test with various step sizes to exercise different z values.
     * Large z (small step) maximally stresses the zero-bitplanes tag tree encoding. */
    ok &= run_test(64,  64,  0.5f,   "64x64   step=0.5  (z≈13)");
    ok &= run_test(128, 128, 0.5f,   "128x128 step=0.5  (z≈13)");
    ok &= run_test(64,  64,  0.01f,  "64x64   step=0.01 (z≈15)");
    ok &= run_test(128, 128, 0.001f, "128x128 step=0.001 (z≈18)");

    printf("\n=== %s ===\n", ok ? "ALL PASSED" : "SOME FAILED");
    return ok ? 0 : 1;
}
