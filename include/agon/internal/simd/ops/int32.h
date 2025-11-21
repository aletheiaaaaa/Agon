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
    inline VecI32<Arch::AVX512> load_i32(const int32_t* ptr) {
        return VecI32<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

    inline void store_i32(int32_t* ptr, const VecI32<Arch::AVX512>& vec) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), vec.data);
    }

    inline VecI32<Arch::AVX512> set1_i32(int32_t val) {
        return VecI32<Arch::AVX512>(_mm512_set1_epi32(val));
    }

    inline VecI32<Arch::AVX512> add_i32(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_add_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX512> sub_i32(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_sub_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX512> mullo_i32(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_mullo_epi32(a.data, b.data));
    }

    inline VecI64<Arch::AVX512> mul_i32(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_mul_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX512> max_i32(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_max_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX512> min_i32(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_min_epi32(a.data, b.data));
    }
#elif defined(__AVX2__)
    inline VecI32<Arch::AVX2> load_i32(const int32_t* ptr) {
        return VecI32<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    inline void store_i32(int32_t* ptr, const VecI32<Arch::AVX2>& vec) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), vec.data);
    }

    inline VecI32<Arch::AVX2> set1_i32(int32_t val) {
        return VecI32<Arch::AVX2>(_mm256_set1_epi32(val));
    }

    inline VecI32<Arch::AVX2> add_i32(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_add_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX2> sub_i32(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_sub_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX2> mullo_i32(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_mullo_epi32(a.data, b.data));
    }

    inline VecI64<Arch::AVX2> mul_i32(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI64<Arch::AVX2>(_mm256_mul_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX2> max_i32(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_max_epi32(a.data, b.data));
    }

    inline VecI32<Arch::AVX2> min_i32(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_min_epi32(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    inline VecI32<Arch::SSE4_1> load_i32(const int32_t* ptr) {
        return VecI32<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    inline void store_i32(int32_t* ptr, const VecI32<Arch::SSE4_1>& vec) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), vec.data);
    }

    inline VecI32<Arch::SSE4_1> set1_i32(int32_t val) {
        return VecI32<Arch::SSE4_1>(_mm_set1_epi32(val));
    }

    inline VecI32<Arch::SSE4_1> add_i32(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_add_epi32(a.data, b.data));
    }

    inline VecI32<Arch::SSE4_1> sub_i32(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_sub_epi32(a.data, b.data));
    }

    inline VecI32<Arch::SSE4_1> mullo_i32(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_mullo_epi32(a.data, b.data));
    }

    inline VecI64<Arch::SSE4_1> mul_i32(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI64<Arch::SSE4_1>(_mm_mul_epi32(a.data, b.data));
    }

    inline VecI32<Arch::SSE4_1> max_i32(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_max_epi32(a.data, b.data));
    }

    inline VecI32<Arch::SSE4_1> min_i32(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_min_epi32(a.data, b.data));
    }
#else
    inline VecI32<Arch::GENERIC> load_i32(const int32_t* ptr) {
        return VecI32<Arch::GENERIC>(*ptr);
    }

    inline void store_i32(int32_t* ptr, const VecI32<Arch::GENERIC>& vec) {
        *ptr = vec.data;
    }

    inline VecI32<Arch::GENERIC> set1_i32(int32_t val) {
        return VecI32<Arch::GENERIC>(val);
    }

    inline VecI32<Arch::GENERIC> add_i32(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data + b.data);
    }

    inline VecI32<Arch::GENERIC> sub_i32(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data - b.data);
    }

    inline VecI32<Arch::GENERIC> mullo_i32(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data * b.data);
    }

    inline VecI64<Arch::GENERIC> mul_i32(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(static_cast<int64_t>(a.data) * static_cast<int64_t>(b.data));
    }

    inline VecI32<Arch::GENERIC> max_i32(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data > b.data ? a.data : b.data);
    }

    inline VecI32<Arch::GENERIC> min_i32(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data < b.data ? a.data : b.data);
    }
#endif
}