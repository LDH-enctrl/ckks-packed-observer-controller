#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <memory>
#include <string>
#include <cstdint>
#include <cmath>
#include <iomanip>

#include <nlohmann/json.hpp>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Parameter.h"
#include "model_obs_params.h"
#include "plant_params.h"


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

    void encode_with_scale(Pt& pt, double value, int level, double scale) {
    std::vector<Complex> msg(1);
    msg[0] = Complex(value, 0.0);

    context->encoder_.Encode(pt, level, scale, msg);
    }

    void encrypt_scalar_with_scale(Ct& ct, double value, int level, double scale) {
        Pt pt;
        encode_with_scale(pt, value, level, scale);
        interface->Encrypt(ct, pt);
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

        const int T = 200;
        int state_level = ckks.param->max_level_;

        std::cout << "Initial CKKS state level: " << state_level << std::endl;

        // Reference plaintext plant/controller
        std::vector<double> xp_ref = {0.1, 0.05, 0.0, 0.0};
        std::vector<double> xc_ref = {0.0, 0.0, 0.0, 0.0};
        std::vector<double> y_ref(2, 0.0);
        double u_ref = 0.0;

        // CKKS-driven plant/controller
        std::vector<double> xp_ckks = {0.1, 0.05, 0.0, 0.0};
        std::vector<double> y_ckks(2, 0.0);
        double u_ckks = 0.0;

        // Encrypted controller state ct_x
        std::vector<Ct> ct_x(4);
        for (int i = 0; i < 4; ++i) {
            ckks.encrypt_scalar(ct_x[i], 0.0, state_level);
        }

        std::ofstream log_file("ckks_closedloop_200.csv");

        log_file << "k,state_level,u_ref,u_ckks,abs_u_err,"
                << "y0_ref,y0_ckks,y0_err,"
                << "y1_ref,y1_ckks,y1_err\n";

        double max_u_err = 0.0;
        int last_valid_step = -1;
        for (int k = 0; k < T; ++k) {
            // ------------------------------------------------------------
            // Reference plaintext plant update with previous u_ref
            // xp_ref = A xp_ref + B u_ref
            // ------------------------------------------------------------
            std::vector<double> xp_ref_next(4, 0.0);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    xp_ref_next[i] += plant_params::A[i][j] * xp_ref[j];
                }
                xp_ref_next[i] += plant_params::B[i][0] * u_ref;
            }

            xp_ref = xp_ref_next;

            // y_ref = C xp_ref
            for (int i = 0; i < 2; ++i) {
                y_ref[i] = 0.0;
                for (int j = 0; j < 4; ++j) {
                    y_ref[i] += plant_params::C[i][j] * xp_ref[j];
                }
            }

            // plaintext observer update:
            // xc_ref_next = F xc_ref + G y_ref
            std::vector<double> xc_ref_next(4, 0.0);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    xc_ref_next[i] += obs_params::F[i][j] * xc_ref[j];
                }
                for (int l = 0; l < 2; ++l) {
                    xc_ref_next[i] += obs_params::G[i][l] * y_ref[l];
                }
            }

            xc_ref = xc_ref_next;

            // u_ref = H xc_ref
            u_ref = 0.0;
            for (int j = 0; j < 4; ++j) {
                u_ref += obs_params::H[0][j] * xc_ref[j];
            }

            // ------------------------------------------------------------
            // CKKS-driven plant update with previous u_ckks
            // ------------------------------------------------------------
            std::vector<double> xp_ckks_next(4, 0.0);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    xp_ckks_next[i] += plant_params::A[i][j] * xp_ckks[j];
                }
                xp_ckks_next[i] += plant_params::B[i][0] * u_ckks;
            }

            xp_ckks = xp_ckks_next;

            // y_ckks = C xp_ckks
            for (int i = 0; i < 2; ++i) {
                y_ckks[i] = 0.0;
                for (int j = 0; j < 4; ++j) {
                    y_ckks[i] += plant_params::C[i][j] * xp_ckks[j];
                }
            }

            // Need at least:
            // state update: state_level -> state_level - 1
            // output:       state_level - 1 -> state_level - 2
            if (state_level <= 2) {
                std::cout << "STOP: insufficient CKKS level before step "
                        << k
                        << ", state_level = "
                        << state_level
                        << std::endl;
                break;
            }

            // Encrypt current plant output at current state_level
            std::vector<Ct> ct_y(2);

            double current_state_scale = ct_x[0].GetScale();

            for (int l = 0; l < 2; ++l) {
                ckks.encrypt_scalar_with_scale(
                    ct_y[l],
                    y_ckks[l],
                    state_level,
                    current_state_scale
                );
}

            // ct_x_next = F ct_x + G ct_y
            std::vector<Ct> ct_x_next(4);

            for (int i = 0; i < 4; ++i) {
                ct_x_next[i] = ckks_linear_state_row(
                    ckks,
                    ct_x,
                    ct_y,
                    i,
                    state_level
                );
            }

            int level_after_state_update = state_level - 1;

            // ct_u = H ct_x_next
            Ct ct_u = ckks_output(
                ckks,
                ct_x_next,
                level_after_state_update
            );

            u_ckks = ckks.decrypt_scalar(ct_u);

            // Update encrypted state
            ct_x = std::move(ct_x_next);
            state_level = level_after_state_update;

            double abs_u_err = std::abs(u_ref - u_ckks);
            double y0_err = std::abs(y_ref[0] - y_ckks[0]);
            double y1_err = std::abs(y_ref[1] - y_ckks[1]);

            max_u_err = std::max(max_u_err, abs_u_err);
            last_valid_step = k;

            log_file << std::setprecision(17)
                    << k << ","
                    << state_level << ","
                    << u_ref << ","
                    << u_ckks << ","
                    << abs_u_err << ","
                    << y_ref[0] << ","
                    << y_ckks[0] << ","
                    << y0_err << ","
                    << y_ref[1] << ","
                    << y_ckks[1] << ","
                    << y1_err
                    << "\n";

            if (k % 5 == 0) {
                std::cout << "k = " << k
                        << ", level = " << state_level
                        << ", |u_err| = " << abs_u_err
                        << std::endl;
            }
        }

        log_file.close();

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Usable steps without bootstrapping: "
                  << last_valid_step + 1
                  << std::endl;
        std::cout << "Last valid step: k = "
                  << last_valid_step
                  << std::endl;
        std::cout << "Final valid state_level: "
                  << state_level
                  << std::endl;
        std::cout << "Max |u_ref - u_ckks|: "
                  << max_u_err
                  << std::endl;
        std::cout << "CSV saved to ckks_closedloop_200.csv"
                  << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}