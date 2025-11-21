#pragma once

#include <cstdint>

#include "../arch.h"
#include "../types.h"

#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE4_1__)
    #include <emmintrin.h>
#endif

namespace agon::simd {
#if defined(__AVX512F__)
    inline VecI16<Arch::AVX512> load_i16(const int16_t* ptr) {
        return VecI16<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

    inline void store_i16(int16_t* ptr, const VecI16<Arch::AVX512>& vec) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), vec.data);
    }

    inline VecI16<Arch::AVX512> set1_i16(int16_t val) {
        return VecI16<Arch::AVX512>(_mm512_set1_epi16(val));
    }

    inline VecI16<Arch::AVX512> add_i16(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_add_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX512> sub_i16(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_sub_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX512> mullo_i16(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_mullo_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX512> mulhi_i16(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_mulhi_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX512> max_i16(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_max_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX512> min_i16(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_min_epi16(a.data, b.data));
    }
#elif defined(__AVX2__)
    inline VecI16<Arch::AVX2> load_i16(const int16_t* ptr) {
        return VecI16<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    inline void store_i16(int16_t* ptr, const VecI16<Arch::AVX2>& vec) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), vec.data);
    }

    inline VecI16<Arch::AVX2> set1_i16(int16_t val) {
        return VecI16<Arch::AVX2>(_mm256_set1_epi16(val));
    }

    inline VecI16<Arch::AVX2> add_i16(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_add_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX2> sub_i16(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_sub_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX2> mullo_i16(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_mullo_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX2> mulhi_i16(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_mulhi_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX2> max_i16(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_max_epi16(a.data, b.data));
    }

    inline VecI16<Arch::AVX2> min_i16(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_min_epi16(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    inline VecI16<Arch::SSE4_1> load_i16(const int16_t* ptr) {
        return VecI16<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    inline void store_i16(int16_t* ptr, const VecI16<Arch::SSE4_1>& vec) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), vec.data);
    }

    inline VecI16<Arch::SSE4_1> set1_i16(int16_t val) {
        return VecI16<Arch::SSE4_1>(_mm_set1_epi16(val));
    }

    inline VecI16<Arch::SSE4_1> add_i16(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_add_epi16(a.data, b.data));
    }

    inline VecI16<Arch::SSE4_1> sub_i16(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_sub_epi16(a.data, b.data));
    }

    inline VecI16<Arch::SSE4_1> mullo_i16(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_mullo_epi16(a.data, b.data));
    }

    inline VecI16<Arch::SSE4_1> mulhi_i16(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_mulhi_epi16(a.data, b.data));
    }

    inline VecI16<Arch::SSE4_1> max_i16(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_max_epi16(a.data, b.data));
    }

    inline VecI16<Arch::SSE4_1> min_i16(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_min_epi16(a.data, b.data));
    }
#else
    inline VecI16<Arch::GENERIC> load_i16(const int16_t* ptr) {
        return VecI16<Arch::GENERIC>(*ptr);
    }

    inline void store_i16(int16_t* ptr, const VecI16<Arch::GENERIC>& vec) {
        *ptr = vec.data;
    }

    inline VecI16<Arch::GENERIC> set1_i16(int16_t val) {
        return VecI16<Arch::GENERIC>(val);
    }

    inline VecI16<Arch::GENERIC> add_i16(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(a.data + b.data);
    }

    inline VecI16<Arch::GENERIC> sub_i16(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(a.data - b.data);
    }

    inline VecI16<Arch::GENERIC> mullo_i16(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(a.data * b.data);
    }

    inline VecI16<Arch::GENERIC> mulhi_i16(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>((static_cast<int32_t>(a.data) * static_cast<int32_t>(b.data)) >> 16));
    }

    inline VecI16<Arch::GENERIC> max_i16(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(a.data > b.data ? a.data : b.data);
    }

    inline VecI16<Arch::GENERIC> min_i16(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(a.data < b.data ? a.data : b.data);
    }
#endif
}
