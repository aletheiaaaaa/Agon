#include "../include/agon/estimator.h"

namespace agon::estim {
    template<typename... Params>
    Estimator::Estimator(Params&... params) : parameters{ &params... }, state{} {}

    Estimator::Estimator(std::initializer_list<ParameterView*> params) : parameters(params), state{} {}

    void Estimator::add_parameter(ParameterView& param) {
        parameters.push_back(&param);
    }
}