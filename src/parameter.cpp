#include "agon/parameter.h"
#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

namespace agon {
  template<typename T>
    requires std::is_floating_point_v<T>
  size_t View<T>::rank() const {
    return shape.size();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  size_t View<T>::numel() const {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  bool View<T>::is_contiguous() const {
    size_t expected_stride = 1;
    for (size_t i = shape.size(); i-- > 0; ) {
      if (strides[i] != expected_stride) return false;
      expected_stride *= shape[i];
    }
    return true;
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  std::vector<T>& View<T>::grad() {
    return ref.get().grad();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  const std::vector<T>& View<T>::grad() const {
    return ref.get().grad();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  std::vector<T>& View<T>::data() {
    return ref.get().data();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  const std::vector<T>& View<T>::data() const {
    return ref.get().data();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  std::vector<T>& Parameter<T>::grad() {
    return grad_;
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  const std::vector<T>& Parameter<T>::grad() const {
    return grad_;
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  std::vector<T>& Parameter<T>::data() {
    return data_;
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  const std::vector<T>& Parameter<T>::data() const {
    return data_;
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  const std::vector<size_t>& Parameter<T>::size() const {
    return shape_;
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  size_t Parameter<T>::size(size_t i) const {
    return shape_[i];
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  size_t Parameter<T>::rank() const {
    return shape_.size();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  size_t Parameter<T>::numel() const {
    return data_.size();
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  void Parameter<T>::zero_grad() {
    std::fill(grad_.begin(), grad_.end(), T(0));
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  void Parameter<T>::accumulate(const std::vector<T>& new_grad) {
    constexpr size_t vec_size = simd::vec<T>::size;
    constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

    size_t i = 0;
    for (; i + vec_size * unroll_factor <= grad_.size(); i += vec_size * unroll_factor) {
      simd::unroll<unroll_factor>([&]<size_t index>() {
        constexpr size_t offset = index * vec_size;

        auto grad_vec = simd::load<T>(&grad_[i + offset]);
        auto new_vec = simd::load<T>(&new_grad[i + offset]);
        grad_vec = simd::add(grad_vec, new_vec);
        simd::store(&grad_[i + offset], grad_vec);
      });
    }

    for (; i < grad_.size(); ++i) {
      grad_[i] += new_grad[i];
    }
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  void Parameter<T>::update(const std::vector<T>& new_val) {
    data_ = new_val;
  }

  template<typename Q, typename T>
    requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
  std::vector<Q> Quantized<Q, T>::quantized() const {
    const auto& vals = this->data();
    std::vector<Q> quantized_data(vals.size());

    T inv_scale = static_cast<T>(1.0f / scale_);
    T zero_point_cast = static_cast<T>(zero_point_);

    constexpr size_t vec_size = simd::vec<T>::size;
    constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

    size_t i = 0;
    for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
      simd::unroll<unroll_factor>([&]<size_t index>() {
        constexpr size_t offset = index * vec_size;

        auto val_vec = simd::load<T>(&vals[i + offset]);
        auto inv_scale_vec = simd::set1<T>(inv_scale);
        auto zero_point_vec = simd::set1<T>(zero_point_cast);
        auto q_vec = simd::fmadd(val_vec, inv_scale_vec, zero_point_vec);

        simd::store(&quantized_data[i + offset], simd::cast<Q>(q_vec));
      });
    }

    for (; i < vals.size(); ++i) {
      quantized_data[i] = static_cast<Q>(inv_scale * vals[i] + zero_point_cast);
    }

    return quantized_data;
  }

  template<typename Q, typename T>
    requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
  std::vector<T> Quantized<Q, T>::fake_quantized() const {
    const auto& vals = this->data();
    std::vector<T> fake_quantized_data(vals.size());

    T inv_scale = static_cast<T>(1.0f / scale_);
    T zero_point_cast = static_cast<T>(zero_point_);

    constexpr size_t vec_size = simd::vec<T>::size;
    constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

    size_t i = 0;
    for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
      simd::unroll<unroll_factor>([&]<size_t index>() {
        constexpr size_t offset = index * vec_size;

        auto val_vec = simd::load<T>(&vals[i + offset]);
        auto scale_vec = simd::set1<T>(scale_);
        auto inv_scale_vec = simd::set1<T>(inv_scale);
        auto zero_point_vec = simd::set1<T>(zero_point_cast);

        auto q_vec = simd::fmadd(val_vec, inv_scale_vec, zero_point_vec);
        q_vec = simd::round(q_vec);

        auto dq_vec = simd::sub(q_vec, zero_point_vec);
        dq_vec = simd::mul(dq_vec, scale_vec);
        simd::store(&fake_quantized_data[i + offset], dq_vec);
      });
    }

    for (; i < vals.size(); ++i) {
      T q = std::round(inv_scale * vals[i] + zero_point_cast);
      fake_quantized_data[i] = scale_ * (q - zero_point_cast);
    }

    return fake_quantized_data;
  }

  template<typename Q, typename T>
    requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
  float Quantized<Q, T>::scale() const {
    return scale_;
  }

  template<typename Q, typename T>
    requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
  float Quantized<Q, T>::zero_point() const {
    return zero_point_;
  }

  template class View<float>;
  template class View<double>;

  template class Parameter<float>;
  template class Parameter<double>;

  template class Quantized<int8_t, float>;
  template class Quantized<int16_t, float>;
  template class Quantized<int8_t, double>;
  template class Quantized<int16_t, double>;
}
