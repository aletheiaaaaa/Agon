#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <typeinfo>
#include <span>
#include <limits>
#include <ranges>
#include <cassert>
#include <numeric>

#include "detail/simd/utils.h"
#include "detail/simd/ops.h"
#include "detail/dedup.h"
#include "detail/flatten.h"

namespace agon {
  struct Slice {
    static constexpr size_t Start = 0;
    static constexpr size_t End = std::numeric_limits<size_t>::max();

    size_t start, end;

    Slice(size_t idx) : start(idx), end(idx) {};
    Slice(size_t idx0, size_t idx1) : start(idx0), end(idx1 - 1) {};
    Slice() : start(Start), end(End) {};
  };

  namespace detail {
    struct ViewParams {
      size_t offset;
      std::vector<size_t> shape;
      std::vector<size_t> strides;
    };

    inline ViewParams compute_view(
      std::span<const Slice> slices,
      std::span<const size_t> src_shape, 
      std::span<const size_t> src_strides
    ) {
      size_t offset = 0;
      std::vector<size_t> shape;
      std::vector<size_t> strides;
      for (const auto& [idx, slice] : std::views::enumerate(slices)) {
        assert((slice.end - 1< src_shape[idx] || slice.end == Slice::End) && "Array index out of bounds");
        assert((slice.start <= slice.end - 1) && "Slice should begin before it ends");
        assert((slices.size() == src_shape.size()) && "There should be as many slices as dimensions");

        offset += slice.start * src_strides[idx];
        size_t dim_size = std::min(slice.end, src_shape[idx]) - slice.start;
        if (dim_size > 0) {
          shape.push_back(dim_size);
          strides.push_back(src_strides[idx]);
        }
      }

      return ViewParams{
        .offset = offset,
        .shape = std::move(shape),
        .strides = std::move(strides)
      };
    }
  }

  template<typename T> 
    requires std::is_floating_point_v<T>
  class Parameter;

  template<typename T>
    requires std::is_floating_point_v<T>
  struct View {
    using is_param_like = std::true_type;
    using DataType = T;

    std::reference_wrapper<Parameter<std::remove_const_t<T>>> ref;
    size_t offset;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    template<typename... Args>
      requires (std::convertible_to<Args, Slice> && ...)
    View<T> operator[](Args... args) {
      std::array<Slice, sizeof...(Args)> slices{args...};

      auto params = detail::compute_view(slices, shape, strides);
      return View<T>{
        .ref = ref,
        .offset = offset + params.offset,
        .shape = std::move(params.shape),
        .strides = std::move(params.strides)
      };
    }

    template<typename... Args>
      requires (std::convertible_to<Args, Slice> && ...)
    const View<const T> operator[](Args... args) const {
      std::array<Slice, sizeof...(Args)> slices{args...};

      auto params = detail::compute_view(slices, shape, strides);
      return View<const T>{
        .ref = std::cref(ref.get()),
        .offset = offset + params.offset,
        .shape = std::move(params.shape),
        .strides = std::move(params.strides)
      };
    }

    std::vector<T>& grad();
    const std::vector<T>& grad() const;

    std::vector<T>& data();
    const std::vector<T>& data() const;

    size_t rank() const;
    size_t numel() const;

    bool is_contiguous() const;
  };

  template<typename T>
    requires std::is_floating_point_v<T>
  class Parameter {
    public:
      using is_param_like = std::true_type;
      using DataType = T;

      explicit Parameter(const std::initializer_list<size_t>& shape) 
        : Parameter(std::vector<size_t>(shape.begin(), shape.end())) {}

      template<typename S>
        requires detail::NestedSpan<S, T>
      explicit Parameter(const S& span) {
        detail::unpack(span, shape_, data_, strides_);
        grad_.resize(data_.size());
      }

      std::vector<T>& grad();
      const std::vector<T>& grad() const;

      std::vector<T>& data();
      const std::vector<T>& data() const;

      const std::vector<size_t>& size() const;
      size_t size(size_t i) const;

      size_t rank() const;
      size_t numel() const;

      template<typename... Args>
        requires (std::convertible_to<Args, Slice> && ...)
      View<T> operator[](Args... args) {
        std::array<Slice, sizeof...(Args)> slices{args...};

        auto params = detail::compute_view(slices, shape_, strides_);
        return View<T>{
          .ref = *this,
          .offset = params.offset,
          .shape = std::move(params.shape),
          .strides = std::move(params.strides)
        };
      }

      template<typename... Args>
        requires (std::convertible_to<Args, Slice> && ...)
      const View<const T> operator[](Args... args) const {
        std::array<Slice, sizeof...(Args)> slices{args...};

        auto params = detail::compute_view(slices, shape_, strides_);
        return View<const T>{
          .ref = *this,
          .offset = params.offset,
          .shape = std::move(params.shape),
          .strides = std::move(params.strides)
        };
      }

      void zero_grad();

      void accumulate(const std::vector<T>& new_grad);
      void update(const std::vector<T>& new_val);

    protected:
      std::vector<size_t> shape_;
      std::vector<size_t> strides_;
      std::vector<T> data_;
      std::vector<T> grad_;

      explicit Parameter(std::vector<size_t>&& shape) : shape_(std::move(shape)) {
        strides_.resize(shape_.size());
        std::exclusive_scan(
          shape_.rbegin(), shape_.rend(), strides_.rbegin(), size_t{1}, std::multiplies<size_t>{}
        );

        size_t num = std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>{});
        data_.resize(num);
        grad_.resize(num);
      }
  };

  template<typename Q, typename T = float>
    requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
  class Quantized : public Parameter<T> {
    public:
      using is_param_like = std::true_type;
      using QuantizedType = Q;

      explicit Quantized(const std::initializer_list<size_t>& shape, float scale = 1.0f, float zero_point = 0.0f) 
        : Parameter<T>(std::vector<size_t>(shape.begin(), shape.end())), scale_(scale), zero_point_(zero_point) {}

      template<typename S>
        requires detail::NestedSpan<S, Q>
      explicit Quantized(const S& span, float scale = 1.0f, float zero_point = 0.0f)
        : Parameter<T>(detail::deduce_shape(span)), scale_(scale), zero_point_(zero_point) {
          detail::fill(span, this->shape_, [&](const auto& leaf, size_t offset) {
            T scale_cast = static_cast<T>(scale);
            T zero_point_cast = static_cast<T>(zero_point);

            constexpr size_t q_vec_size = simd::vec<Q>::size;
            constexpr size_t f_vec_size = simd::vec<T>::size;
            constexpr size_t unroll_factor = std::max(simd::UNROLL_FACTOR / 2, 1);
            constexpr size_t slice_idx = q_vec_size / f_vec_size;

            int i = 0;
            for (; i + q_vec_size * unroll_factor <= leaf.size(); i += q_vec_size * unroll_factor) {
              simd::unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset0 = index * q_vec_size;

                auto scale_vec = simd::set1<T>(scale_cast);
                auto zero_point_vec = simd::set1<T>(zero_point_cast);

                auto q_vec = simd::load<Q>(&leaf[i + offset0]);
                simd::unroll<slice_idx>([&]<size_t slice>() {
                  constexpr size_t offset1 = slice * f_vec_size;

                  auto q_float_vec = simd::cast<T, slice>(q_vec);

                  auto val_vec = simd::sub(q_float_vec, zero_point_vec);
                  val_vec = simd::mul(val_vec, scale_vec);

                  simd::store(&this->data_[offset + i + offset0 + offset1], val_vec);
                });
              });
            }

            for (; i < leaf.size(); ++i) {
              this->data_[offset + i] = scale_cast * (static_cast<T>(leaf[i]) - zero_point_cast);
            }
          });
        }

      std::vector<Q> quantized() const;
      std::vector<T> fake_quantized() const;

      float scale() const;
      float zero_point() const;

    private:
      float scale_ = 1.0f;
      float zero_point_ = 0.0f;
  };

  template<typename T>
  concept ParamLike = T::is_param_like::value;

  template<typename T>
  using AsParameter_t = Parameter<typename T::DataType>;

  template<typename T>
  using RefVec = std::vector<std::reference_wrapper<T>>;

  template<typename DedupedTuple>
  struct ParameterPack {
    detail::TransformTuple_t<RefVec, DedupedTuple> data{};

    template<typename... Ts>
      requires (std::derived_from<Ts, Parameter<typename Ts::DataType>> && ...)
    ParameterPack(Ts&... params) {
      (std::get<RefVec<AsParameter_t<Ts>>>(data)
        .emplace_back(static_cast<AsParameter_t<Ts>&>(params)), ...);
    }

    template<typename T>
      requires std::derived_from<T, Parameter<typename T::DataType>>
    void add_parameter(T& param) {
      std::get<RefVec<AsParameter_t<T>>>(data)
        .emplace_back(static_cast<AsParameter_t<T>&>(param));
    }
  };
  template<typename... Ts>
  ParameterPack(Ts&...) -> ParameterPack<detail::Canonicalized_t<AsParameter_t<std::decay_t<Ts>>...>>;

  template<typename T>
  struct ExtractType {};
  template<typename T>
  struct ExtractType<Parameter<T>> {
    using Type = T;
  };
  template<typename T>
  using ExtractType_t = typename ExtractType<T>::Type;
}