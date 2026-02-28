#include "agon/optimizers/impls/lion.h"

#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

#include <fstream>
#include <filesystem>

namespace agon::optim {
  template<typename... Ts>
  void Lion<Ts...>::step() {
    std::apply([&](auto&... param_vecs) {
      (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
        auto& param = param_ref.get();
        using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

        auto& grad_full = param.grad();
        auto& data_full = param.data();
        auto& mom_full = std::get<std::vector<T>>(state_.momentum);

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
          simd::unroll<unroll_factor>([&]<size_t index>(){
            constexpr size_t offset = index * vec_size;

            auto grad = simd::load<T>(&grad_full[i + offset]);
            auto mom = simd::load<T>(&mom_full[i + offset]);

            if (options_.maximize) grad = simd::neg(grad);

            auto beta1 = simd::set1<T>(options_.beta1);
            auto update = simd::fmadd(beta1, mom, grad);
            update = simd::fnmadd(beta1, grad, mom);

            auto data = simd::load<T>(&data_full[i + offset]);

            if (options_.lambda) update = simd::fnmadd(simd::set1<T>(options_.lambda), data, update);
            data = simd::fmadd(simd::set1<T>(options_.lr), simd::sign(update), data);
            simd::store(&data_full[i + offset], data);

            auto beta2 = simd::set1<T>(options_.beta2);
            mom = simd::fmadd(beta2, mom, grad);
            mom = simd::fnmadd(beta2, grad, mom);
            simd::store(&mom_full[i + offset], mom);
          });
        }

        for (; i < grad_full.size(); ++i) {
          T grad = options_.maximize ? -grad_full[i] : grad_full[i];
          T mom = options_.beta1 * mom_full[i] + (1 - options_.beta1) * grad;

          T update = std::copysign(options_.lr, mom);
          if (options_.lambda) update = -options_.lambda * data_full[i] + update;

          data_full[i] += update;
          mom_full[i] = options_.beta2 * mom_full[i] + (1 - options_.beta2) * grad;
        }
      }), ...);
    }, this->parameters_.data);

    state_.step++;
  }

  template<typename... Ts>
  void Lion<Ts...>::load_from_bin(const std::string& path_str) {
    std::filesystem::path path(path_str);
    if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path_str);

    in.read(reinterpret_cast<char*>(&options_), sizeof(options_));
    in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

    std::apply([&](auto&... param_vecs) {
      (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
        auto& param = param_ref.get();
        using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
        auto& mom = std::get<std::vector<T>>(state_.momentum);

        in.read(reinterpret_cast<char*>(&param.data()), param.numel() * sizeof(T));
        in.read(reinterpret_cast<char*>(mom.data()), param.numel() * sizeof(T));
      }), ...);
    }, this->parameters_.data);
  }

  template<typename... Ts>
  void Lion<Ts...>::save_to_bin(const std::string& path_str) const {
    std::filesystem::path path(path_str);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file: " + path_str);

    out.write(reinterpret_cast<const char*>(&options_), sizeof(options_));
    out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

    std::apply([&](auto&... param_vecs) {
      (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
        auto& param = param_ref.get();
        using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
        auto& mom = std::get<std::vector<T>>(state_.momentum);

        out.write(reinterpret_cast<const char*>(&param.data()), param.numel() * sizeof(T));
        out.write(reinterpret_cast<const char*>(mom.data()), param.numel() * sizeof(T));
      }), ...);
    }, this->parameters_.data);
  }

  template class Lion<std::tuple<Parameter<float>>>;
  template class Lion<std::tuple<Parameter<double>>>;
  template class Lion<std::tuple<Parameter<float>, Parameter<double>>>;
}