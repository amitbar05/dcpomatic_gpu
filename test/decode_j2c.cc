/*
    Decode a J2K codestream with OpenJPEG to verify it decodes cleanly.
    Build:
      g++ -std=c++17 -O2 test/decode_j2c.cc -lopenjp2 -o test/decode_j2c
    Usage:
      ./test/decode_j2c /tmp/gpu_correct.j2c
*/
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <openjpeg-2.5/openjpeg.h>

struct MemBuf {
    const uint8_t* data;
    size_t size;
    size_t pos;
};

static OPJ_SIZE_T mem_read(void* dst, OPJ_SIZE_T n, void* ud) {
    auto* mb = static_cast<MemBuf*>(ud);
    if (mb->pos >= mb->size) return (OPJ_SIZE_T) -1;
    size_t avail = mb->size - mb->pos;
    size_t take = n < avail ? n : avail;
    memcpy(dst, mb->data + mb->pos, take);
    mb->pos += take;
    return take;
}

static OPJ_OFF_T mem_skip(OPJ_OFF_T n, void* ud) {
    auto* mb = static_cast<MemBuf*>(ud);
    OPJ_OFF_T avail = static_cast<OPJ_OFF_T>(mb->size) - static_cast<OPJ_OFF_T>(mb->pos);
    OPJ_OFF_T take = (n < avail) ? n : avail;
    mb->pos += take;
    return take;
}

static OPJ_BOOL mem_seek(OPJ_OFF_T n, void* ud) {
    auto* mb = static_cast<MemBuf*>(ud);
    if (n > static_cast<OPJ_OFF_T>(mb->size)) return OPJ_FALSE;
    mb->pos = n;
    return OPJ_TRUE;
}

static void info_cb(const char* msg, void*) { printf("[info] %s", msg); }
static void warn_cb(const char* msg, void*) { printf("[warn] %s", msg); }
static void err_cb (const char* msg, void*) { printf("[err ] %s", msg); }

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s file.j2c\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 2; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(sz);
    fread(data.data(), 1, sz, f);
    fclose(f);
    printf("Read %ld bytes from %s\n", sz, argv[1]);

    MemBuf mb { data.data(), static_cast<size_t>(sz), 0 };
    opj_stream_t* str = opj_stream_default_create(OPJ_TRUE);
    opj_stream_set_read_function  (str, mem_read);
    opj_stream_set_skip_function  (str, mem_skip);
    opj_stream_set_seek_function  (str, mem_seek);
    opj_stream_set_user_data      (str, &mb, nullptr);
    opj_stream_set_user_data_length(str, sz);

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_info_handler (codec, info_cb, nullptr);
    opj_set_warning_handler(codec, warn_cb, nullptr);
    opj_set_error_handler(codec, err_cb,  nullptr);

    opj_dparameters_t dp;
    opj_set_default_decoder_parameters(&dp);
    if (!opj_setup_decoder(codec, &dp)) { fprintf(stderr, "setup failed\n"); return 3; }

    opj_image_t* img = nullptr;
    if (!opj_read_header(str, codec, &img)) { fprintf(stderr, "read_header failed\n"); return 4; }
    printf("Header OK: %ux%u, %u components\n", img->x1 - img->x0, img->y1 - img->y0, img->numcomps);

    if (!opj_decode(codec, str, img)) { fprintf(stderr, "decode failed\n"); return 5; }
    if (!opj_end_decompress(codec, str)) { fprintf(stderr, "end_decompress failed\n"); return 6; }

    /* Sanity: first component stats */
    OPJ_INT32* d = img->comps[0].data;
    uint32_t w = img->comps[0].w, h = img->comps[0].h;
    long long sum = 0, sq = 0, mn = d[0], mx = d[0];
    for (uint32_t i = 0; i < w * h; i++) {
        sum += d[i];
        sq  += static_cast<long long>(d[i]) * d[i];
        if (d[i] < mn) mn = d[i];
        if (d[i] > mx) mx = d[i];
    }
    double mean = sum / static_cast<double>(w * h);
    printf("Component 0: %ux%u, mean=%.1f range=[%lld, %lld]\n", w, h, mean, mn, mx);

    opj_image_destroy(img);
    opj_destroy_codec(codec);
    opj_stream_destroy(str);
    printf("DECODE OK\n");
    return 0;
}
