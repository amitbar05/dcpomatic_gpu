/*
    Run libdcp's verify_j2k on a J2C file to find the exact failure mode
    that produces "A picture frame has invalid JPEG2000 codestream".

    Build:
      g++ -std=c++17 -O2 test/verify_dcp_j2k.cc \
          -Ideps/install/include/libdcp-1.0 \
          -Ldeps/install/lib64 -ldcp-1.0 \
          -Wl,-rpath,deps/install/lib64 \
          -o test/verify_dcp_j2k

    Usage:
      ./test/verify_dcp_j2k /tmp/gpu_correct.j2c
*/

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <dcp/array_data.h>
#include <dcp/verify.h>
#include <dcp/verify_j2k.h>

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s file.j2c\n", argv[0]); return 1; }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::cout << "Read " << raw.size() << " bytes\n";

    auto data = std::make_shared<dcp::ArrayData>(
        reinterpret_cast<uint8_t const*>(raw.data()),
        static_cast<int>(raw.size()));

    std::vector<dcp::VerificationNote> notes;
    dcp::verify_j2k(data, 0, 0, 24, notes);

    std::cout << "verify_j2k produced " << notes.size() << " notes:\n";
    for (auto const& n : notes) {
        std::cout << "  [type=" << int(n.type())
                  << " code=" << int(n.code());
        if (n.note())  std::cout << " note=\"" << *n.note() << "\"";
        if (n.frame()) std::cout << " frame=" << *n.frame();
        std::cout << "]\n";
    }
    return notes.empty() ? 0 : 3;
}
