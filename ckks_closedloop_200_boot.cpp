#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Parameter.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"

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

using BootCtx = cheddar::BootContext<Word>;
using BootCtxPtr = std::shared_ptr<BootCtx>;
using EvkRequest = cheddar::EvkRequest;

using Clock = std::chrono::steady_clock;

static double elapsed_ms(
    const Clock::time_point& t0,
    const Clock::time_point& t1
) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct CheddarBoot {
    std::unique_ptr<Param> param;
    CtxPtr context;
    BootCtxPtr boot_context;
    std::unique_ptr<UI> interface;

    int log_degree = 0;
    int default_encryption_level = 0;
    int boot_num_slots = 1 << 15;  // 32768, same as official Bootstrapping.cpp

    explicit CheddarBoot(const std::string& json_path) {
        std::ifstream json_file(json_path);

        if (!json_file.is_open()) {
            throw std::runtime_error("Failed to open parameter file: " + json_path);
        }

        json data = json::parse(json_file);
        json_file.close();

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
            level_config.emplace_back(
                pair[0].get<int>(),
                pair[1].get<int>()
            );
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

        bool enable_boot = false;
        if (data.contains("boot")) {
            enable_boot = data["boot"].get<bool>();
        }

        if (!enable_boot) {
            throw std::runtime_error("This parameter file does not enable bootstrapping.");
        }

        int num_cts_levels = data["num_cts_levels"].get<int>();
        int num_stc_levels = data["num_stc_levels"].get<int>();

        std::cout << "json_path = " << json_path << std::endl;
        std::cout << "log_degree = " << log_degree << std::endl;
        std::cout << "default_encryption_level = "
                  << default_encryption_level << std::endl;
        std::cout << "num_cts_levels = " << num_cts_levels << std::endl;
        std::cout << "num_stc_levels = " << num_stc_levels << std::endl;
        std::cout << "main_primes.size = " << main_primes.size() << std::endl;
        std::cout << "aux_primes.size = " << aux_primes.size() << std::endl;

        boot_context = BootCtx::Create(
            *param,
            cheddar::BootParameter(
                param->max_level_,
                num_cts_levels,
                num_stc_levels
            )
        );

        context = boot_context;
        interface = std::make_unique<UI>(context);

        std::cout << "param->max_level_ = "
                  << param->max_level_
                  << std::endl;

        std::cout << "Preparing bootstrapping..." << std::endl;
        std::cout << "boot_num_slots = " << boot_num_slots << std::endl;

        boot_context->PrepareEvalMod();
        boot_context->PrepareEvalSpecialFFT(boot_num_slots);

        EvkRequest req;
        boot_context->AddRequiredRotations(req, boot_num_slots);
        interface->PrepareRotationKey(req);

        std::cout << "Bootstrapping preparation done." << std::endl;
    }

    double scale_at_level(int level) const {
        if (level <= default_encryption_level) {
            return param->GetScale(level);
        }

        return param->GetRescalePrimeProd(level);
    }

    void encode_scalar(Pt& pt, double value, int level) {
        // Use full slot vector to match bootstrapping slot count.
        std::vector<Complex> msg(boot_num_slots, Complex(value, 0.0));
        double scale = scale_at_level(level);
        context->encoder_.Encode(pt, level, scale, msg);
    }

    void encode_scalar_with_scale(Pt& pt, double value, int level, double scale) {
        std::vector<Complex> msg(boot_num_slots, Complex(value, 0.0));
        context->encoder_.Encode(pt, level, scale, msg);
    }

    void encode_constant(Const& c, double value, int level) {
        double scale = scale_at_level(level);
        context->encoder_.EncodeConstant(c, level, scale, value);
    }

    void encrypt_scalar(Ct& ct, double value, int level) {
        Pt pt;
        encode_scalar(pt, value, level);
        interface->Encrypt(ct, pt);
    }

    void encrypt_scalar_with_scale(Ct& ct, double value, int level, double scale) {
        Pt pt;
        encode_scalar_with_scale(pt, value, level, scale);
        interface->Encrypt(ct, pt);
    }

    double decrypt_scalar(const Ct& ct) {
        Pt pt;
        interface->Decrypt(pt, ct);

        std::vector<Complex> msg;
        context->encoder_.Decode(msg, pt);

        return msg[0].real();
    }

    int level_of(const Ct& ct) const {
        return param->NPToLevel(ct.GetNP());
    }

    void multiply_const_rescale(Ct& out, const Ct& in, double c, int level) {
        Const cc;
        encode_constant(cc, c, level);

        Ct tmp;
        context->Mult(tmp, in, cc);
        context->Rescale(out, tmp);
    }

    void level_down_to_zero(Ct& ct, int& level) {
        while (level > 0) {
            Ct tmp;
            multiply_const_rescale(tmp, ct, 1.0, level);
            ct = std::move(tmp);
            level -= 1;
        }
    }

    void bootstrap_basic(Ct& out, const Ct& in) {
        boot_context->Boot(out, in, interface->GetEvkMap());
    }
};

static std::string param_path(const std::string& name) {
    return std::string(std::getenv("HOME")) +
           "/Cheddar_project/cheddar-fhe/parameters/" +
           name;
}

static void clear_cuda_error_state() {
    cudaGetLastError();
}

static void try_dummy_boot_warmup() {
    std::cout << "\n=== Dummy boot warm-up with bootparam_30.json ==="
              << std::endl;

    try {
        {
            CheddarBoot dummy(param_path("bootparam_30.json"));
        }

        std::cout << "Dummy warm-up unexpectedly succeeded." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Dummy warm-up failed/caught as expected:" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    clear_cuda_error_state();

    std::cout << "Continue to actual closed-loop simulation." << std::endl;
}

Ct ckks_linear_state_row(
    CheddarBoot& ckks,
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
            term.SetScale(acc.GetScale());

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
            term.SetScale(acc.GetScale());

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
    CheddarBoot& ckks,
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
            term.SetScale(acc.GetScale());

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
        try_dummy_boot_warmup();

        std::cout << "\n=== Actual closed-loop with bootparam_35.json ==="
                  << std::endl;

        CheddarBoot ckks(param_path("bootparam_35.json"));

        const int T = 200;
        int state_level = ckks.default_encryption_level;

        std::cout << "Initial CKKS state level selected for controller: "
                  << state_level
                  << std::endl;
        std::cout << "Boot output/default level: "
                  << ckks.default_encryption_level
                  << std::endl;

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

        std::ofstream log_file("ckks_closedloop_200_boot.csv");

        log_file << "k,state_level,boot_count,u_ref,u_ckks,abs_u_err,"
                 << "y0_ref,y0_ckks,y0_err,"
                 << "y1_ref,y1_ckks,y1_err,"
                 << "loop_ms,ckks_control_ms,encrypt_y_ms,state_hom_ms,"
                 << "output_hom_ms,decrypt_u_ms,boot_total_ms,boot_pure_ms\n";

        double max_u_err = 0.0;
        int last_valid_step = -1;
        int boot_count = 0;

        double total_loop_ms = 0.0;
        double total_ckks_control_ms = 0.0;
        double total_encrypt_y_ms = 0.0;
        double total_state_hom_ms = 0.0;
        double total_output_hom_ms = 0.0;
        double total_decrypt_u_ms = 0.0;
        double total_boot_total_ms = 0.0;
        double total_boot_pure_ms = 0.0;

        double max_loop_ms = 0.0;
        double max_ckks_control_ms = 0.0;
        double max_boot_total_ms = 0.0;
        double max_boot_pure_ms = 0.0;

        for (int k = 0; k < T; ++k) {
            auto loop_t0 = Clock::now();

            double loop_ms = 0.0;
            double ckks_control_ms = 0.0;
            double encrypt_y_ms = 0.0;
            double state_hom_ms = 0.0;
            double output_hom_ms = 0.0;
            double decrypt_u_ms = 0.0;
            double boot_total_ms = 0.0;
            double boot_pure_ms = 0.0;
            // ------------------------------------------------------------
            // Reference plaintext plant update with previous u_ref
            // ------------------------------------------------------------
            std::vector<double> xp_ref_next(4, 0.0);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    xp_ref_next[i] += plant_params::A[i][j] * xp_ref[j];
                }
                xp_ref_next[i] += plant_params::B[i][0] * u_ref;
            }

            xp_ref = xp_ref_next;

            for (int i = 0; i < 2; ++i) {
                y_ref[i] = 0.0;
                for (int j = 0; j < 4; ++j) {
                    y_ref[i] += plant_params::C[i][j] * xp_ref[j];
                }
            }

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

            for (int i = 0; i < 2; ++i) {
                y_ckks[i] = 0.0;
                for (int j = 0; j < 4; ++j) {
                    y_ckks[i] += plant_params::C[i][j] * xp_ckks[j];
                }
            }

            // ------------------------------------------------------------
            // Bootstrap encrypted controller state if level is low.
            // Need to lower to level 0 first to match Cheddar boot test.
            // ------------------------------------------------------------
            if (state_level <= 4) {
                auto boot_total_t0 = Clock::now();

                std::cout << "\nBOOT trigger at step "
                          << k
                          << ", state_level = "
                          << state_level
                          << std::endl;

                for (int i = 0; i < 4; ++i) {
                    int working_level = ckks.level_of(ct_x[i]);

                    auto level_down_t0 = Clock::now();
                    ckks.level_down_to_zero(ct_x[i], working_level);
                    double level_down_ms = elapsed_ms(level_down_t0, Clock::now());

                    Ct refreshed;

                    auto boot_ct_t0 = Clock::now();
                    ckks.bootstrap_basic(refreshed, ct_x[i]);
                    double boot_ct_ms = elapsed_ms(boot_ct_t0, Clock::now());

                    boot_pure_ms += boot_ct_ms;

                    ct_x[i] = std::move(refreshed);

                    std::cout << "  booted state index "
                              << i
                              << ", actual_level = "
                              << ckks.level_of(ct_x[i])
                              << ", scale = "
                              << ct_x[i].GetScale()
                              << ", level_down_ms = "
                              << level_down_ms
                              << ", boot_ct_ms = "
                              << boot_ct_ms
                              << std::endl;
                }

                state_level = ckks.level_of(ct_x[0]);

                for (int i = 1; i < 4; ++i) {
                    int li = ckks.level_of(ct_x[i]);

                    if (li != state_level) {
                        throw std::runtime_error(
                            "Booted state ciphertext levels are inconsistent."
                        );
                    }

                    ct_x[i].SetScale(ct_x[0].GetScale());
                }

                boot_count += 1;

                boot_total_ms = elapsed_ms(boot_total_t0, Clock::now());

                total_boot_total_ms += boot_total_ms;
                total_boot_pure_ms += boot_pure_ms;
                max_boot_total_ms = std::max(max_boot_total_ms, boot_total_ms);
                max_boot_pure_ms = std::max(max_boot_pure_ms, boot_pure_ms);

                std::cout << "BOOT done. New state_level = "
                          << state_level
                          << ", boot_count = "
                          << boot_count
                          << ", boot_total_ms = "
                          << boot_total_ms
                          << ", boot_pure_ms = "
                          << boot_pure_ms
                          << std::endl;
            }

            if (state_level <= 1) {
                std::cout << "STOP: insufficient CKKS level before step "
                          << k
                          << ", state_level = "
                          << state_level
                          << std::endl;
                break;
            }

            // Encrypt current plant output at current state level and scale
            auto encrypt_y_t0 = Clock::now();

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

            encrypt_y_ms = elapsed_ms(encrypt_y_t0, Clock::now());

            // ct_x_next = F ct_x + G ct_y
            auto state_hom_t0 = Clock::now();

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

            state_hom_ms = elapsed_ms(state_hom_t0, Clock::now());

            int level_after_state_update = state_level - 1;

            // ct_u = H ct_x_next
            auto output_hom_t0 = Clock::now();

            Ct ct_u = ckks_output(
                ckks,
                ct_x_next,
                level_after_state_update
            );

            output_hom_ms = elapsed_ms(output_hom_t0, Clock::now());

            auto decrypt_u_t0 = Clock::now();
            u_ckks = ckks.decrypt_scalar(ct_u);
            decrypt_u_ms = elapsed_ms(decrypt_u_t0, Clock::now());

            // Update encrypted state
            ct_x = std::move(ct_x_next);
            state_level = level_after_state_update;

            double abs_u_err = std::abs(u_ref - u_ckks);
            double y0_err = std::abs(y_ref[0] - y_ckks[0]);
            double y1_err = std::abs(y_ref[1] - y_ckks[1]);

            max_u_err = std::max(max_u_err, abs_u_err);
            last_valid_step = k;

            ckks_control_ms =
                encrypt_y_ms + state_hom_ms + output_hom_ms + decrypt_u_ms;

            loop_ms = elapsed_ms(loop_t0, Clock::now());

            total_loop_ms += loop_ms;
            total_ckks_control_ms += ckks_control_ms;
            total_encrypt_y_ms += encrypt_y_ms;
            total_state_hom_ms += state_hom_ms;
            total_output_hom_ms += output_hom_ms;
            total_decrypt_u_ms += decrypt_u_ms;

            max_loop_ms = std::max(max_loop_ms, loop_ms);
            max_ckks_control_ms = std::max(max_ckks_control_ms, ckks_control_ms);

            log_file << std::setprecision(17)
                     << k << ","
                     << state_level << ","
                     << boot_count << ","
                     << u_ref << ","
                     << u_ckks << ","
                     << abs_u_err << ","
                     << y_ref[0] << ","
                     << y_ckks[0] << ","
                     << y0_err << ","
                     << y_ref[1] << ","
                     << y_ckks[1] << ","
                     << y1_err << ","
                     << loop_ms << ","
                     << ckks_control_ms << ","
                     << encrypt_y_ms << ","
                     << state_hom_ms << ","
                     << output_hom_ms << ","
                     << decrypt_u_ms << ","
                     << boot_total_ms << ","
                     << boot_pure_ms
                     << "\n";

            if (k % 5 == 0) {
                std::cout << "k = " << k
                          << ", level = " << state_level
                          << ", boots = " << boot_count
                          << ", |u_err| = " << abs_u_err
                          << ", loop_ms = " << loop_ms
                          << ", ckks_control_ms = " << ckks_control_ms
                          << std::endl;
            }
        }

        log_file.close();

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Requested steps: "
                  << T
                  << std::endl;
        std::cout << "Completed valid steps: "
                  << last_valid_step + 1
                  << std::endl;
        std::cout << "Last valid step: k = "
                  << last_valid_step
                  << std::endl;
        std::cout << "Final state_level: "
                  << state_level
                  << std::endl;
        std::cout << "Boot count: "
                  << boot_count
                  << std::endl;
        std::cout << "Max |u_ref - u_ckks|: "
                  << max_u_err
                  << std::endl;

        int completed_steps = last_valid_step + 1;

        if (completed_steps > 0) {
            std::cout << "Avg loop time [ms]: "
                      << total_loop_ms / completed_steps
                      << std::endl;
            std::cout << "Max loop time [ms]: "
                      << max_loop_ms
                      << std::endl;

            std::cout << "Avg CKKS control time without boot [ms]: "
                      << total_ckks_control_ms / completed_steps
                      << std::endl;
            std::cout << "Max CKKS control time without boot [ms]: "
                      << max_ckks_control_ms
                      << std::endl;

            std::cout << "Avg encrypt y time [ms]: "
                      << total_encrypt_y_ms / completed_steps
                      << std::endl;
            std::cout << "Avg encrypted state update time [ms]: "
                      << total_state_hom_ms / completed_steps
                      << std::endl;
            std::cout << "Avg encrypted output time [ms]: "
                      << total_output_hom_ms / completed_steps
                      << std::endl;
            std::cout << "Avg decrypt u time [ms]: "
                      << total_decrypt_u_ms / completed_steps
                      << std::endl;
        }

        if (boot_count > 0) {
            std::cout << "Total boot trigger time [ms]: "
                      << total_boot_total_ms
                      << std::endl;
            std::cout << "Total pure Boot() time [ms]: "
                      << total_boot_pure_ms
                      << std::endl;
            std::cout << "Avg boot trigger time per refresh [ms]: "
                      << total_boot_total_ms / boot_count
                      << std::endl;
            std::cout << "Avg pure Boot() time per refresh [ms]: "
                      << total_boot_pure_ms / boot_count
                      << std::endl;
            std::cout << "Avg pure Boot() time per ciphertext [ms]: "
                      << total_boot_pure_ms / (4.0 * boot_count)
                      << std::endl;
            std::cout << "Max boot trigger time [ms]: "
                      << max_boot_total_ms
                      << std::endl;
            std::cout << "Max pure Boot() time per refresh [ms]: "
                      << max_boot_pure_ms
                      << std::endl;
        }

        std::cout << "CSV saved to ckks_closedloop_200_boot.csv"
                  << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
