#pragma once

#include <stdfloat>

#include "../arch.h"
#include "../types.h"

#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE2__)
    #include <emmintrin.h>
#endif

namespace agon::simd {
#if defined(__AVX512F__) && defined(HAS_FLOAT16) && HAS_FLOAT16
    inline VecF16<Arch::AVX512> load_f16(const std::float16_t* ptr) {
        return VecF16<Arch::AVX512>(_mm512_loadu_ph(ptr));
    }

    inline void store_f16(std::float16_t* ptr, const VecF16<Arch::AVX512>& vec) {
        _mm512_storeu_ph(ptr, vec.data);
    }

    inline VecF16<Arch::AVX512> set1_f16(std::float16_t val) {
        return VecF16<Arch::AVX512>(_mm512_set1_ph(val));
    }

    inline VecF16<Arch::AVX512> add_f16(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_add_ph(a.data, b.data));
    }

    inline VecF16<Arch::AVX512> sub_f16(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_sub_ph(a.data, b.data));
    }

    inline VecF16<Arch::AVX512> mul_f16(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_mul_ph(a.data, b.data));
    }

    inline VecF16<Arch::AVX512> div_f16(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_div_ph(a.data, b.data));
    }

    inline VecF16<Arch::AVX512> max_f16(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_max_ph(a.data, b.data));
    }

    inline VecF16<Arch::AVX512> min_f16(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_min_ph(a.data, b.data));
    }
#endif
}