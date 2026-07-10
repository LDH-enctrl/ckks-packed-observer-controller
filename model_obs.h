#pragma once

#include <vector>
#include "model_obs_params.h"

class obs_plain {
private:
    std::vector<double> x = {0.0, 0.0, 0.0, 0.0};
    double u = 0.0;

public:
    obs_plain() = default;

    void state_update(const std::vector<double>& y) {
        std::vector<double> x_next(4, 0.0);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                x_next[i] += obs_params::F[i][j] * x[j];
            }

            for (int l = 0; l < 2; ++l) {
                x_next[i] += obs_params::G[i][l] * y[l];
            }
        }

        x = x_next;
    }

    double get_output() {
        u = 0.0;

        for (int j = 0; j < 4; ++j) {
            u += obs_params::H[0][j] * x[j];
        }

        return u;
    }

    const std::vector<double>& get_state() const {
        return x;
    }
};
