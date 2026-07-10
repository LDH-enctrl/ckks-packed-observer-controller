#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <memory>
#include <string>
#include <cstdint>
#include <cmath>

#include <nlohmann/json.hpp>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Parameter.h"
#include "model_obs_params.h"


using json = nlohmann::json;
using Word = std::uint32_t;
using Complex = cheddar::Complex;

using Ct = cheddar::Ciphertext<Word>;
using Pt = cheddar::Plaintext<Word>;
using Const = cheddar::Constant<Word>;
using Param = cheddar::Parameter<Word>;
using Ctx = cheddar::Context<Word>;
using CtxPtr = cheddar::ContextPtr<Word>;
using UI = cheddar::UserInterface<Word>;

struct CheddarSimple {
    std::unique_ptr<Param> param;
    CtxPtr context;
    std::unique_ptr<UI> interface;

    int log_degree = 0;
    int default_encryption_level = 0;

    explicit CheddarSimple(const std::string& json_path) {
        std::ifstream json_file(json_path);
        if (!json_file.is_open()) {
            throw std::runtime_error("Failed to open parameter file: " + json_path);
        }

        json data = json::parse(json_file);

        log_degree = data["log_degree"];
        int log_default_scale = data["log_default_scale"];
        double default_scale = static_cast<double>(UINT64_C(1) << log_default_scale);
        default_encryption_level = data["default_encryption_level"];

        std::vector<Word> main_primes;
        for (const auto& p : data["main_primes"]) {
            main_primes.push_back(p.get<Word>());
        }

        std::vector<Word> aux_primes;
        for (const auto& p : data["auxiliary_primes"]) {
            aux_primes.push_back(p.get<Word>());
        }

        std::vector<Word> ter_primes;
        if (data.contains("terminal_primes")) {
            for (const auto& p : data["terminal_primes"]) {
                ter_primes.push_back(p.get<Word>());
            }
        }

        std::vector<std::pair<int, int>> level_config;
        for (const auto& pair : data["level_config"]) {
            level_config.emplace_back(pair[0].get<int>(), pair[1].get<int>());
        }

        std::pair<int, int> additional_base = {0, 0};
        if (data.contains("additional_base")) {
            additional_base = {
                data["additional_base"][0].get<int>(),
                data["additional_base"][1].get<int>()
            };
        }

        param = std::make_unique<Param>(
            log_degree,
            default_scale,
            default_encryption_level,
            level_config,
            main_primes,
            aux_primes,
            ter_primes,
            additional_base
        );

        if (data.contains("dense_hamming_weight")) {
            param->SetDenseHammingWeight(data["dense_hamming_weight"].get<int>());
        }

        if (data.contains("sparse_hamming_weight")) {
            param->SetSparseHammingWeight(data["sparse_hamming_weight"].get<int>());
        }

        context = Ctx::Create(*param);
        interface = std::make_unique<UI>(context);
    }

    double scale_at_level(int level) const {
        if (level <= default_encryption_level) {
            return param->GetScale(level);
        }
        return param->GetRescalePrimeProd(level);
    }

    void encode(Pt& pt, double value, int level) {
        std::vector<Complex> msg(1);
        msg[0] = Complex(value, 0.0);

        double scale = scale_at_level(level);
        context->encoder_.Encode(pt, level, scale, msg);
    }

    void encode_constant(Const& c, double value, int level) {
        double scale = scale_at_level(level);
        context->encoder_.EncodeConstant(c, level, scale, value);
    }

    void encrypt_scalar(Ct& ct, double value, int level) {
        Pt pt;
        encode(pt, value, level);
        interface->Encrypt(ct, pt);
    }

    double decrypt_scalar(const Ct& ct) {
        Pt pt;
        interface->Decrypt(pt, ct);

        std::vector<Complex> msg;
        context->encoder_.Decode(msg, pt);

        return msg[0].real();
    }

    void multiply_const_rescale(Ct& out, const Ct& in, double c, int level) {
        Const cc;
        encode_constant(cc, c, level);

        Ct tmp;
        context->Mult(tmp, in, cc);
        context->Rescale(out, tmp);
    }
};

Ct ckks_linear_state_row(
    CheddarSimple& ckks,
    const std::vector<Ct>& ct_x,
    const std::vector<Ct>& ct_y,
    int row,
    int input_level
) {
    if (input_level <= 0) {
        throw std::runtime_error("input_level must be positive for rescale.");
    }

    Ct acc;
    bool initialized = false;

    for (int j = 0; j < 4; ++j) {
        double coeff = obs_params::F[row][j];

        if (std::abs(coeff) < 1e-15) {
            continue;
        }

        Ct term;
        ckks.multiply_const_rescale(term, ct_x[j], coeff, input_level);

        if (!initialized) {
            acc = std::move(term);
            initialized = true;
        }
        else {
            Ct tmp;
            ckks.context->Add(tmp, acc, term);
            acc = std::move(tmp);
        }
    }

    for (int l = 0; l < 2; ++l) {
        double coeff = obs_params::G[row][l];

        if (std::abs(coeff) < 1e-15) {
            continue;
        }

        Ct term;
        ckks.multiply_const_rescale(term, ct_y[l], coeff, input_level);

        if (!initialized) {
            acc = std::move(term);
            initialized = true;
        }
        else {
            Ct tmp;
            ckks.context->Add(tmp, acc, term);
            acc = std::move(tmp);
        }
    }

    if (!initialized) {
        ckks.encrypt_scalar(acc, 0.0, input_level - 1);
    }

    return acc;
}

Ct ckks_output(
    CheddarSimple& ckks,
    const std::vector<Ct>& ct_x_next,
    int input_level
) {
    if (input_level <= 0) {
        throw std::runtime_error("input_level must be positive for rescale.");
    }

    Ct acc;
    bool initialized = false;

    for (int j = 0; j < 4; ++j) {
        double coeff = obs_params::H[0][j];

        if (std::abs(coeff) < 1e-15) {
            continue;
        }

        Ct term;
        ckks.multiply_const_rescale(term, ct_x_next[j], coeff, input_level);

        if (!initialized) {
            acc = std::move(term);
            initialized = true;
        }
        else {
            Ct tmp;
            ckks.context->Add(tmp, acc, term);
            acc = std::move(tmp);
        }
    }

    if (!initialized) {
        ckks.encrypt_scalar(acc, 0.0, input_level - 1);
    }

    return acc;
}

int main() {
    try {
        std::string param_path =
            std::string(std::getenv("HOME")) +
            "/Cheddar_project/cheddar-fhe/parameters/bootparam_30.json";

        CheddarSimple ckks(param_path);

        int level = ckks.param->max_level_;

        std::cout << "Initial CKKS level: " << level << std::endl;

        // 임의의 observer state와 plant output.
        // 나중에 QLabs에서는 이 x는 암호화된 controller state,
        // y는 매 step plant output을 암호화한 값이 됩니다.
        std::vector<double> x_plain = {
            0.01,
            -0.02,
            0.03,
            -0.04
        };

        std::vector<double> y_plain = {
            0.05,
            -0.03
        };

        // Plaintext observer one-step:
        // x_next = F x + G y
        std::vector<double> x_next_plain(4, 0.0);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                x_next_plain[i] += obs_params::F[i][j] * x_plain[j];
            }

            for (int l = 0; l < 2; ++l) {
                x_next_plain[i] += obs_params::G[i][l] * y_plain[l];
            }
        }

        // u = H x_next
        double u_plain = 0.0;

        for (int j = 0; j < 4; ++j) {
            u_plain += obs_params::H[0][j] * x_next_plain[j];
        }

        // Encrypt x and y
        std::vector<Ct> ct_x(4);
        std::vector<Ct> ct_y(2);

        for (int j = 0; j < 4; ++j) {
            ckks.encrypt_scalar(ct_x[j], x_plain[j], level);
        }

        for (int l = 0; l < 2; ++l) {
            ckks.encrypt_scalar(ct_y[l], y_plain[l], level);
        }

        // CKKS observer one-step:
        // ct_x_next = F ct_x + G ct_y
        std::vector<Ct> ct_x_next(4);

        for (int i = 0; i < 4; ++i) {
            ct_x_next[i] = ckks_linear_state_row(
                ckks,
                ct_x,
                ct_y,
                i,
                level
            );
        }

        int level_after_state_update = level - 1;

        // ct_u = H ct_x_next
        Ct ct_u = ckks_output(
            ckks,
            ct_x_next,
            level_after_state_update
        );

        // Decrypt results
        std::vector<double> x_next_ckks(4, 0.0);

        for (int i = 0; i < 4; ++i) {
            x_next_ckks[i] = ckks.decrypt_scalar(ct_x_next[i]);
        }

        double u_ckks = ckks.decrypt_scalar(ct_u);

        std::cout << "\n=== x_next comparison ===" << std::endl;

        for (int i = 0; i < 4; ++i) {
            double err = std::abs(x_next_plain[i] - x_next_ckks[i]);

            std::cout << "x_next[" << i << "] plain = "
                      << x_next_plain[i]
                      << ", ckks = "
                      << x_next_ckks[i]
                      << ", abs error = "
                      << err
                      << std::endl;
        }

        std::cout << "\n=== u comparison ===" << std::endl;
        std::cout << "u_plain = " << u_plain << std::endl;
        std::cout << "u_ckks  = " << u_ckks << std::endl;
        std::cout << "abs error = " << std::abs(u_plain - u_ckks) << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}