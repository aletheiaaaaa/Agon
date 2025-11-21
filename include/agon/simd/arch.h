#include <cstdint>

namespace agon::simd {
    enum class Arch : uint8_t {
        GENERIC,
        SSE2,
        AVX2,
        AVX512,
    };

    inline constexpr Arch detect_arch() {
#if defined(__AVX512F__)
        return Arch::AVX512;
#elif defined(__AVX2__)
        return Arch::AVX2;
#elif defined(__SSE2__)
        return Arch::SSE2;
#else
        return Arch::GENERIC;
#endif
    }

    constexpr Arch CURRENT_ARCH = detect_arch();
}

#if defined(__AVX512FP16__)
    #define HAS_FLOAT16 1
#else
    #define HAS_FLOAT16 0
#endif

#if defined(__AVX512BF16__)
    #define HAS_BFLOAT16 1
#else
    #define HAS_BFLOAT16 0
#endif
