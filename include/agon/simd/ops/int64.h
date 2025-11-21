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
    #include <smmintrin.h>
#endif

namespace agon::simd {
#if defined(__AVX512F__)
    inline VecI64<Arch::AVX512> load_i64(const int64_t* ptr) {
        return VecI64<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

    inline void store_i64(int64_t* ptr, const VecI64<Arch::AVX512>& vec) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), vec.data);
    }

    inline VecI64<Arch::AVX512> set1_i64(int64_t val) {
        return VecI64<Arch::AVX512>(_mm512_set1_epi64(val));
    }

    inline VecI64<Arch::AVX512> add_i64(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_add_epi64(a.data, b.data));
    }

    inline VecI64<Arch::AVX512> sub_i64(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_sub_epi64(a.data, b.data));
    }

#if defined(__AVX512DQ__)
    inline VecI64<Arch::AVX512> mullo_i64(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_mullo_epi64(a.data, b.data));
    }
#endif

    inline VecI64<Arch::AVX512> max_i64(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_max_epi64(a.data, b.data));
    }

    inline VecI64<Arch::AVX512> min_i64(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_min_epi64(a.data, b.data));
    }
#elif defined(__AVX2__)
    inline VecI64<Arch::AVX2> load_i64(const int64_t* ptr) {
        return VecI64<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    inline void store_i64(int64_t* ptr, const VecI64<Arch::AVX2>& vec) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), vec.data);
    }

    inline VecI64<Arch::AVX2> set1_i64(int64_t val) {
        return VecI64<Arch::AVX2>(_mm256_set1_epi64x(val));
    }

    inline VecI64<Arch::AVX2> add_i64(const VecI64<Arch::AVX2>& a, const VecI64<Arch::AVX2>& b) {
        return VecI64<Arch::AVX2>(_mm256_add_epi64(a.data, b.data));
    }

    inline VecI64<Arch::AVX2> sub_i64(const VecI64<Arch::AVX2>& a, const VecI64<Arch::AVX2>& b) {
        return VecI64<Arch::AVX2>(_mm256_sub_epi64(a.data, b.data));
    }

    inline VecI64<Arch::AVX2> max_i64(const VecI64<Arch::AVX2>& a, const VecI64<Arch::AVX2>& b) {
        __m256i cmp = _mm256_cmpgt_epi64(a.data, b.data);
        return VecI64<Arch::AVX2>(_mm256_blendv_epi8(b.data, a.data, cmp));
    }

    inline VecI64<Arch::AVX2> min_i64(const VecI64<Arch::AVX2>& a, const VecI64<Arch::AVX2>& b) {
        __m256i cmp = _mm256_cmpgt_epi64(a.data, b.data);
        return VecI64<Arch::AVX2>(_mm256_blendv_epi8(a.data, b.data, cmp));
    }
#elif defined(__SSE4_1__)
    inline VecI64<Arch::SSE4_1> load_i64(const int64_t* ptr) {
        return VecI64<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    inline void store_i64(int64_t* ptr, const VecI64<Arch::SSE4_1>& vec) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), vec.data);
    }

    inline VecI64<Arch::SSE4_1> set1_i64(int64_t val) {
        return VecI64<Arch::SSE4_1>(_mm_set1_epi64x(val));
    }

    inline VecI64<Arch::SSE4_1> add_i64(const VecI64<Arch::SSE4_1>& a, const VecI64<Arch::SSE4_1>& b) {
        return VecI64<Arch::SSE4_1>(_mm_add_epi64(a.data, b.data));
    }

    inline VecI64<Arch::SSE4_1> sub_i64(const VecI64<Arch::SSE4_1>& a, const VecI64<Arch::SSE4_1>& b) {
        return VecI64<Arch::SSE4_1>(_mm_sub_epi64(a.data, b.data));
    }

    inline VecI64<Arch::SSE4_1> max_i64(const VecI64<Arch::SSE4_1>& a, const VecI64<Arch::SSE4_1>& b) {
        __m128i cmp = _mm_cmpgt_epi64(a.data, b.data);
        return VecI64<Arch::SSE4_1>(_mm_blendv_epi8(b.data, a.data, cmp));
    }

    inline VecI64<Arch::SSE4_1> min_i64(const VecI64<Arch::SSE4_1>& a, const VecI64<Arch::SSE4_1>& b) {
        __m128i cmp = _mm_cmpgt_epi64(a.data, b.data);
        return VecI64<Arch::SSE4_1>(_mm_blendv_epi8(a.data, b.data, cmp));
    }
#else
    inline VecI64<Arch::GENERIC> load_i64(const int64_t* ptr) {
        return VecI64<Arch::GENERIC>(*ptr);
    }

    inline void store_i64(int64_t* ptr, const VecI64<Arch::GENERIC>& vec) {
        *ptr = vec.data;
    }

    inline VecI64<Arch::GENERIC> set1_i64(int64_t val) {
        return VecI64<Arch::GENERIC>(val);
    }

    inline VecI64<Arch::GENERIC> add_i64(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data + b.data);
    }

    inline VecI64<Arch::GENERIC> sub_i64(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data - b.data);
    }

    inline VecI64<Arch::GENERIC> mullo_i64(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data * b.data);
    }

    inline VecI64<Arch::GENERIC> max_i64(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data > b.data ? a.data : b.data);
    }

    inline VecI64<Arch::GENERIC> min_i64(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data < b.data ? a.data : b.data);
    }
#endif
}
