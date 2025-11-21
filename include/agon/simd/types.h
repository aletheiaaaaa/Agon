#include <cstdint>

#include "arch.h"

#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE2__)
    #include <emmintrin.h>
#endif

namespace agon::simd {
    template<Arch arch>
    struct VecI8;

    template<Arch arch>
    struct VecI16;

    template<Arch arch>
    struct VecI32;

    template<Arch arch>
    struct VecI64;

    template<Arch arch>
    struct VecBF16;

    template<Arch arch>
    struct VecF16;

    template<Arch arch>
    struct VecF32;

    template<Arch arch>
    struct VecF64;

#if defined(__AVX512F__)
    template<>
    struct VecI8<Arch::AVX512> {
        static constexpr size_t size = 64;
        __m512i data;

        VecI8() = default;
        explicit VecI8(__m512i val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::AVX512> {
        static constexpr size_t size = 32;
        __m512i data;

        VecI16() = default;
        explicit VecI16(__m512i val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::AVX512> {
        static constexpr size_t size = 16;
        __m512i data;

        VecI32() = default;
        explicit VecI32(__m512i val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::AVX512> {
        static constexpr size_t size = 8;
        __m512i data;

        VecI64() = default;
        explicit VecI64(__m512i val) : data(val) {}
    };

#if HAS_FLOAT16
    template<>
    struct VecF16<Arch::AVX512> {
        static constexpr size_t size = 32;
        __m512h data;

        VecF16() = default;
        explicit VecF16(__m512h val) : data(val) {}
    };
#endif

#if HAS_BFLOAT16
    template<>
    struct VecBF16<Arch::AVX512> {
        static constexpr size_t size = 32;
        __m512bh data;

        VecBF16() = default;
        explicit VecBF16(__m512bh val) : data(val) {}
    };
#endif

    template<>
    struct VecF32<Arch::AVX512> {
        static constexpr size_t size = 16;
        __m512 data;

        VecF32() = default;
        explicit VecF32(__m512 val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::AVX512> {
        static constexpr size_t size = 8;
        __m512d data;

        VecF64() = default;
        explicit VecF64(__m512d val) : data(val) {}
    };
#elif defined(__AVX2__)
    template<>
    struct VecI8<Arch::AVX2> {
        static constexpr size_t size = 32;
        __m256i data;

        VecI8() = default;
        explicit VecI8(__m256i val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::AVX2> {
        static constexpr size_t size = 16;
        __m256i data;

        VecI16() = default;
        explicit VecI16(__m256i val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::AVX2> {
        static constexpr size_t size = 8;
        __m256i data;

        VecI32() = default;
        explicit VecI32(__m256i val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::AVX2> {
        static constexpr size_t size = 4;
        __m256i data;

        VecI64() = default;
        explicit VecI64(__m256i val) : data(val) {}
    };

    template<>
    struct VecF32<Arch::AVX2> {
        static constexpr size_t size = 8;
        __m256 data;

        VecF32() = default;
        explicit VecF32(__m256 val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::AVX2> {
        static constexpr size_t size = 4;
        __m256d data;

        VecF64() = default;
        explicit VecF64(__m256d val) : data(val) {}
    };
#elif defined(__SSE2__)
    template<>
    struct VecI8<Arch::SSE2> {
        static constexpr size_t size = 16;
        __m128i data;

        VecI8() = default;
        explicit VecI8(__m128i val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::SSE2> {
        static constexpr size_t size = 8;
        __m128i data;

        VecI16() = default;
        explicit VecI16(__m128i val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::SSE2> {
        static constexpr size_t size = 4;
        __m128i data;

        VecI32() = default;
        explicit VecI32(__m128i val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::SSE2> {
        static constexpr size_t size = 2;
        __m128i data;

        VecI64() = default;
        explicit VecI64(__m128i val) : data(val) {}
    };

    template<>
    struct VecF32<Arch::SSE2> {
        static constexpr size_t size = 4;
        __m128 data;

        VecF32() = default;
        explicit VecF32(__m128 val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::SSE2> {
        static constexpr size_t size = 2;
        __m128d data;

        VecF64() = default;
        explicit VecF64(__m128d val) : data(val) {}
    };
#else
    template<>
    struct VecI8<Arch::GENERIC> {
        static constexpr size_t size = 1;
        int8_t data;

        VecI8() = default;
        explicit VecI8(int8_t val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::GENERIC> {
        static constexpr size_t size = 1;
        int16_t data;

        VecI16() = default;
        explicit VecI16(int16_t val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::GENERIC> {
        static constexpr size_t size = 1;
        int32_t data;

        VecI32() = default;
        explicit VecI32(int32_t val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::GENERIC> {
        static constexpr size_t size = 1;
        int64_t data;

        VecI64() = default;
        explicit VecI64(int64_t val) : data(val) {}
    };

    template<>
    struct VecF32<Arch::GENERIC> {
        static constexpr size_t size = 1;
        float data;

        VecF32() = default;
        explicit VecF32(float val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::GENERIC> {
        static constexpr size_t size = 1;
        double data;

        VecF64() = default;
        explicit VecF64(double val) : data(val) {}
    };
#endif

    using vec_i8 = VecI8<CURRENT_ARCH>;
    using vec_i16 = VecI16<CURRENT_ARCH>;
    using vec_i32 = VecI32<CURRENT_ARCH>;
    using vec_i64 = VecI64<CURRENT_ARCH>;
    using vec_f32 = VecF32<CURRENT_ARCH>;
    using vec_f64 = VecF64<CURRENT_ARCH>;

#if HAS_FLOAT16
    using vec_f16 = VecF16<CURRENT_ARCH>;
#endif

#if HAS_BFLOAT16
    using vec_bf16 = VecBF16<CURRENT_ARCH>;
#endif
}