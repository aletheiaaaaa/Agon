#pragma once

#include "parameter.h"

#include <vector>

namespace agon::estim {
    struct EstimatorState {
        size_t func_calls = 0;
    };

    class Estimator {
        public:
            template<typename... Params>
            explicit Estimator(Params&... params);
            explicit Estimator(std::initializer_list<ParameterView*> params);

            void add_parameter(ParameterView& param);

            virtual void needs_eval() = 0;
            virtual void perturb() = 0;
            virtual void observe() = 0;
            virtual void finalize() = 0;

            virtual ~Estimator() = default;
        protected:
            EstimatorState state;
            std::vector<ParameterView*> parameters;
    };
}