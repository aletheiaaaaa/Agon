#pragma once

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
    inline VecF32<Arch::AVX512> load_f32(const float* ptr) {
        return VecF32<Arch::AVX512>(_mm512_loadu_ps(ptr));
    }

    inline void store_f32(float* ptr, const VecF32<Arch::AVX512>& vec) {
        _mm512_storeu_ps(ptr, vec.data);
    }

    inline VecF32<Arch::AVX512> set1_f32(float val) {
        return VecF32<Arch::AVX512>(_mm512_set1_ps(val));
    }

    inline VecF32<Arch::AVX512> add_f32(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_add_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX512> sub_f32(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_sub_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX512> mul_f32(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_mul_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX512> div_f32(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_div_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX512> max_f32(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_max_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX512> min_f32(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_min_ps(a.data, b.data));
    }
#elif defined(__AVX2__)
    inline VecF32<Arch::AVX2> load_f32(const float* ptr) {
        return VecF32<Arch::AVX2>(_mm256_loadu_ps(ptr));
    }

    inline void store_f32(float* ptr, const VecF32<Arch::AVX2>& vec) {
        _mm256_storeu_ps(ptr, vec.data);
    }

    inline VecF32<Arch::AVX2> set1_f32(float val) {
        return VecF32<Arch::AVX2>(_mm256_set1_ps(val));
    }

    inline VecF32<Arch::AVX2> add_f32(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_add_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX2> sub_f32(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_sub_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX2> mul_f32(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_mul_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX2> div_f32(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_div_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX2> max_f32(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_max_ps(a.data, b.data));
    }

    inline VecF32<Arch::AVX2> min_f32(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_min_ps(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    inline VecF32<Arch::SSE4_1> load_f32(const float* ptr) {
        return VecF32<Arch::SSE4_1>(_mm_loadu_ps(ptr));
    }

    inline void store_f32(float* ptr, const VecF32<Arch::SSE4_1>& vec) {
        _mm_storeu_ps(ptr, vec.data);
    }

    inline VecF32<Arch::SSE4_1> set1_f32(float val) {
        return VecF32<Arch::SSE4_1>(_mm_set1_ps(val));
    }

    inline VecF32<Arch::SSE4_1> add_f32(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_add_ps(a.data, b.data));
    }

    inline VecF32<Arch::SSE4_1> sub_f32(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_sub_ps(a.data, b.data));
    }

    inline VecF32<Arch::SSE4_1> mul_f32(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_mul_ps(a.data, b.data));
    }

    inline VecF32<Arch::SSE4_1> div_f32(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_div_ps(a.data, b.data));
    }

    inline VecF32<Arch::SSE4_1> max_f32(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_max_ps(a.data, b.data));
    }

    inline VecF32<Arch::SSE4_1> min_f32(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_min_ps(a.data, b.data));
    }
#else
    inline VecF32<Arch::GENERIC> load_f32(const float* ptr) {
        return VecF32<Arch::GENERIC>(*ptr);
    }

    inline void store_f32(float* ptr, const VecF32<Arch::GENERIC>& vec) {
        *ptr = vec.data;
    }

    inline VecF32<Arch::GENERIC> set1_f32(float val) {
        return VecF32<Arch::GENERIC>(val);
    }

    inline VecF32<Arch::GENERIC> add_f32(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data + b.data);
    }

    inline VecF32<Arch::GENERIC> sub_f32(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data - b.data);
    }

    inline VecF32<Arch::GENERIC> mul_f32(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data * b.data);
    }

    inline VecF32<Arch::GENERIC> div_f32(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data / b.data);
    }

    inline VecF32<Arch::GENERIC> max_f32(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data > b.data ? a.data : b.data);
    }

    inline VecF32<Arch::GENERIC> min_f32(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data < b.data ? a.data : b.data);
    }
#endif
}