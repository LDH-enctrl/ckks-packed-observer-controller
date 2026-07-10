#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/EvkRequest.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"

#include "model_obs_params.h"
#include "plant_params.h"

using json = nlohmann::json;

using Word = std::uint32_t;
using Complex = cheddar::Complex;

using Ct = cheddar::Ciphertext<Word>;
using Pt = cheddar::Plaintext<Word>;
using Param = cheddar::Parameter<Word>;
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

using Vec4 = std::array<double, 4>;
using Mat4 = std::array<std::array<double, 4>, 4>;

static bool near_zero_mask(const Vec4& v) {
    for (double x : v) {
        if (std::abs(x) > 1e-15) {
            return false;
        }
    }
    return true;
}

static std::string param_path(const std::string& name) {
    return std::string(std::getenv("HOME")) +
           "/Cheddar_project/cheddar-fhe/parameters/" +
           name;
}

static void clear_cuda_error_state() {
    cudaGetLastError();
}

struct CheddarBootPacked {
    std::unique_ptr<Param> param;
    CtxPtr context;
    BootCtxPtr boot_context;
    std::unique_ptr<UI> interface;

    int log_degree = 0;
    int default_encryption_level = 0;
    int boot_num_slots = 1 << 15;  // 32768, same as Cheddar boot test.

    bool positive_rotation_is_left = true;

    explicit CheddarBootPacked(const std::string& json_path) {
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

        // Rotation keys 1, 2, and 3 are already included in the
        // bootstrapping rotation-key request.
        //
        // Additional right rotations are represented modulo S:
        //    -1 -> S-1, -2 -> S-2, -3 -> S-3.
        //
        // IMPORTANT:
        // Explicitly pass default_encryption_level. In Cheddar,
        // max_level == -1 is a special Dense-to-Sparse key base and
        // must not be used for these ordinary rotation keys.
        const std::array<int, 3> wrap_rotations = {
            boot_num_slots - 3,
            boot_num_slots - 2,
            boot_num_slots - 1
        };

        for (int rot : wrap_rotations) {
            interface->PrepareRotationKey(
                rot,
                default_encryption_level
            );
        }

        std::cout << "Bootstrapping preparation done." << std::endl;

        detect_rotation_orientation();
    }

    int level_of(const Ct& ct) const {
        return param->NPToLevel(ct.GetNP());
    }

    double scale_at_level(int level) const {
        if (level <= default_encryption_level) {
            return param->GetScale(level);
        }

        return param->GetRescalePrimeProd(level);
    }

    std::vector<Complex> make_periodic4_message(const Vec4& values) const {
        // Despite the legacy function name, only the first four slots
        // are used. This avoids relying on approximate equality between
        // independently encoded repeated slots.
        std::vector<Complex> msg(
            boot_num_slots,
            Complex(0.0, 0.0)
        );

        for (int i = 0; i < 4; ++i) {
            msg[i] = Complex(values[i], 0.0);
        }

        return msg;
    }

    void encode_periodic4(
        Pt& pt,
        const Vec4& values,
        int level,
        double scale
    ) {
        auto msg = make_periodic4_message(values);
        context->encoder_.Encode(pt, level, scale, msg);
    }

    void encrypt_periodic4(
        Ct& ct,
        const Vec4& values,
        int level,
        double scale
    ) {
        Pt pt;
        encode_periodic4(pt, values, level, scale);
        interface->Encrypt(ct, pt);
    }

    void encrypt_periodic4(
        Ct& ct,
        const Vec4& values,
        int level
    ) {
        encrypt_periodic4(ct, values, level, scale_at_level(level));
    }

    std::vector<Complex> decrypt_vector(const Ct& ct) {
        Pt pt;
        interface->Decrypt(pt, ct);

        std::vector<Complex> msg;
        context->encoder_.Decode(msg, pt);

        return msg;
    }

    Vec4 decrypt_first4(const Ct& ct) {
        auto msg = decrypt_vector(ct);

        Vec4 out{};
        for (int i = 0; i < 4; ++i) {
            out[i] = msg[i].real();
        }

        return out;
    }

    double decrypt_h_terms_sum4(const Ct& ct) {
        auto msg = decrypt_vector(ct);

        return msg[0].real()
             + msg[1].real()
             + msg[2].real()
             + msg[3].real();
    }

    int normalize_rotation_index(int d) const {
        int normalized = d % boot_num_slots;

        if (normalized < 0) {
            normalized += boot_num_slots;
        }

        return normalized;
    }

    void rotate_pos(Ct& out, const Ct& in, int d) {
        int rot_idx = normalize_rotation_index(d);

        if (rot_idx == 0) {
            context->Copy(out, in);
            return;
        }

        context->HRot(
            out,
            in,
            interface->GetRotationKey(rot_idx),
            rot_idx
        );
    }

    void detect_rotation_orientation() {
        const int level = default_encryption_level;
        Vec4 test = {0.0, 1.0, 2.0, 3.0};

        Ct ct;
        encrypt_periodic4(ct, test, level);

        Ct rot;
        rotate_pos(rot, ct, 1);

        Vec4 got = decrypt_first4(rot);

        double left_err =
            std::abs(got[0] - 1.0) +
            std::abs(got[1] - 2.0) +
            std::abs(got[2] - 3.0) +
            std::abs(got[3] - 0.0);

        // With only slots 0..3 populated, a global right rotation
        // produces approximately [0, 0, 1, 2] in the first four slots.
        double right_err =
            std::abs(got[0] - 0.0) +
            std::abs(got[1] - 0.0) +
            std::abs(got[2] - 1.0) +
            std::abs(got[3] - 2.0);

        positive_rotation_is_left = (left_err <= right_err);

        std::cout << "Rotation +1 first4 = ["
                  << got[0] << ", "
                  << got[1] << ", "
                  << got[2] << ", "
                  << got[3] << "]"
                  << std::endl;

        std::cout << "Detected positive HRot orientation: "
                  << (positive_rotation_is_left ? "left" : "right")
                  << std::endl;
    }

    Vec4 diagonal_mask_for_rotation(
        const Mat4& A,
        int d
    ) const {
        Vec4 mask{};

        for (int row = 0; row < 4; ++row) {
            int col = 0;

            if (positive_rotation_is_left) {
                col = (row + d) & 3;
            }
            else {
                col = (row - d + 4) & 3;
            }

            mask[row] = A[row][col];
        }

        return mask;
    }

    void multiply_plain4_rescale(
        Ct& out,
        const Ct& in,
        const Vec4& mask
    ) {
        int level = level_of(in);

        Pt pt;
        encode_periodic4(pt, mask, level, in.GetScale());

        Ct tmp;
        context->Mult(tmp, in, pt);
        context->Rescale(out, tmp);
    }

    void rotate_logical_left(
        Ct& out,
        const Ct& in,
        int logical_shift
    ) {
        if (logical_shift == 0) {
            context->Copy(out, in);
            return;
        }

        // Convert a logical left rotation into Cheddar's detected
        // physical HRot direction.
        int physical_shift =
            positive_rotation_is_left
                ? logical_shift
                : -logical_shift;

        rotate_pos(out, in, physical_shift);
    }

    void accumulate_matvec_term(
        Ct& acc,
        bool& initialized,
        const Ct& in,
        int logical_shift,
        const Vec4& mask
    ) {
        if (near_zero_mask(mask)) {
            return;
        }

        Ct shifted;

        if (logical_shift == 0) {
            context->Copy(shifted, in);
        }
        else {
            rotate_logical_left(shifted, in, logical_shift);
        }

        Ct term;
        multiply_plain4_rescale(term, shifted, mask);

        if (!initialized) {
            acc = std::move(term);
            initialized = true;
            return;
        }

        term.SetScale(acc.GetScale());

        Ct tmp;
        context->Add(tmp, acc, term);
        acc = std::move(tmp);
    }

    void packed_matvec4(
        Ct& out,
        const Mat4& A,
        const Ct& in
    ) {
        int input_level = level_of(in);

        if (input_level <= 0) {
            throw std::runtime_error(
                "packed_matvec4 requires input level > 0."
            );
        }

        Ct acc;
        bool initialized = false;

        // We want, for each diagonal d,
        //
        //   shifted[i] = x[(i+d) mod 4].
        //
        // A global CKKS rotation does not wrap at slot 4, so each
        // diagonal is split into:
        //
        //   main part: no 4-slot boundary crossing
        //   wrap part: values crossing the 4-slot boundary
        //
        // Example for d=1:
        //   RotLeft(x, 1)  -> [x1,x2,x3,0,...]
        //   RotLeft(x,-3)  -> [...,x0 at slot 3,...]
        for (int d = 0; d < 4; ++d) {
            Vec4 main_mask{0.0, 0.0, 0.0, 0.0};
            Vec4 wrap_mask{0.0, 0.0, 0.0, 0.0};

            for (int row = 0; row < 4; ++row) {
                int col_without_wrap = row + d;

                if (col_without_wrap < 4) {
                    main_mask[row] = A[row][col_without_wrap];
                }
                else {
                    wrap_mask[row] = A[row][col_without_wrap - 4];
                }
            }

            // Non-wrapped positions use logical left rotation by d.
            accumulate_matvec_term(
                acc,
                initialized,
                in,
                d,
                main_mask
            );

            // Wrapped positions use the equivalent negative rotation:
            //
            // d=1 -> -3
            // d=2 -> -2
            // d=3 -> -1
            if (d > 0) {
                accumulate_matvec_term(
                    acc,
                    initialized,
                    in,
                    d - 4,
                    wrap_mask
                );
            }
        }

        if (!initialized) {
            encrypt_periodic4(
                acc,
                Vec4{0.0, 0.0, 0.0, 0.0},
                input_level - 1
            );
        }

        out = std::move(acc);
    }

    void packed_h_terms(
        Ct& out,
        const Vec4& h,
        const Ct& x
    ) {
        multiply_plain4_rescale(out, x, h);
    }

    void level_down_to_zero(Ct& ct) {
        int level = level_of(ct);

        if (level <= 0) {
            return;
        }

        Ct tmp;
        context->LevelDown(tmp, ct, 0);
        ct = std::move(tmp);
    }

    void bootstrap_basic(Ct& out, const Ct& in) {
        boot_context->Boot(out, in, interface->GetEvkMap());
    }
};

static Mat4 make_F() {
    Mat4 F{};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            F[i][j] = obs_params::F[i][j];
        }
    }

    return F;
}

static Mat4 make_Gbar() {
    Mat4 G{};

    for (int i = 0; i < 4; ++i) {
        G[i][0] = obs_params::G[i][0];
        G[i][1] = obs_params::G[i][1];
        G[i][2] = 0.0;
        G[i][3] = 0.0;
    }

    return G;
}

static Vec4 make_Hmask() {
    return Vec4{
        obs_params::H[0][0],
        obs_params::H[0][1],
        obs_params::H[0][2],
        obs_params::H[0][3]
    };
}

static void try_dummy_boot_warmup() {
    std::cout << "\n=== Dummy boot warm-up with bootparam_30.json ==="
              << std::endl;

    try {
        {
            CheddarBootPacked dummy(param_path("bootparam_30.json"));
        }

        std::cout << "Dummy warm-up unexpectedly succeeded." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Dummy warm-up failed/caught as expected:" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    clear_cuda_error_state();

    std::cout << "Continue to actual packed closed-loop simulation." << std::endl;
}

int main() {
    try {
        try_dummy_boot_warmup();

        std::cout << "\n=== Packed CKKS closed-loop with bootparam_35.json ==="
                  << std::endl;

        CheddarBootPacked ckks(param_path("bootparam_35.json"));

        const int T = 200;

        Mat4 F = make_F();
        Mat4 Gbar = make_Gbar();
        Vec4 Hmask = make_Hmask();

        int state_level = ckks.default_encryption_level;

        std::cout << "Initial packed CKKS state level selected for controller: "
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

        // Plaintext controller driven by the same y_ckks.
        // This isolates packed-CKKS controller error from plant/reference error.
        std::vector<double> xc_ckks_plain = {0.0, 0.0, 0.0, 0.0};
        double u_ckks_plain = 0.0;

        // Packed encrypted controller state:
        // [x0, x1, x2, x3, 0, 0, 0, 0, ...]
        Ct ct_x;
        ckks.encrypt_periodic4(
            ct_x,
            Vec4{0.0, 0.0, 0.0, 0.0},
            state_level
        );

        // ------------------------------------------------------------
        // Optional packed algebra self-test.
        // Run with:
        //   PACKED_SELF_TEST=1 ./build/ckks_closedloop_200_boot_packed
        // ------------------------------------------------------------
        if (std::getenv("PACKED_SELF_TEST") != nullptr) {
            std::cout << "\n=== PACKED ALGEBRA SELF TEST ===" << std::endl;

            Vec4 x_test{0.1, -0.2, 0.3, -0.4};
            Vec4 y_test{0.05, -0.03, 0.0, 0.0};

            Ct ct_x_test;
            Ct ct_y_test;

            ckks.encrypt_periodic4(
                ct_x_test,
                x_test,
                state_level
            );

            ckks.encrypt_periodic4(
                ct_y_test,
                y_test,
                state_level,
                ct_x_test.GetScale()
            );

            Ct ct_Fx_test;
            Ct ct_Gy_test;
            Ct ct_xnext_test;

            ckks.packed_matvec4(ct_Fx_test, F, ct_x_test);
            ckks.packed_matvec4(ct_Gy_test, Gbar, ct_y_test);

            ct_Gy_test.SetScale(ct_Fx_test.GetScale());
            ckks.context->Add(ct_xnext_test, ct_Fx_test, ct_Gy_test);

            Ct ct_u_terms_test;
            ckks.packed_h_terms(ct_u_terms_test, Hmask, ct_xnext_test);

            Vec4 Fx_dec = ckks.decrypt_first4(ct_Fx_test);
            Vec4 Gy_dec = ckks.decrypt_first4(ct_Gy_test);
            Vec4 xnext_dec = ckks.decrypt_first4(ct_xnext_test);

            auto u_terms_msg = ckks.decrypt_vector(ct_u_terms_test);

            Vec4 Fx_plain{};
            Vec4 Gy_plain{};
            Vec4 xnext_plain{};

            for (int i = 0; i < 4; ++i) {
                Fx_plain[i] = 0.0;
                Gy_plain[i] = 0.0;

                for (int j = 0; j < 4; ++j) {
                    Fx_plain[i] += obs_params::F[i][j] * x_test[j];
                }

                for (int j = 0; j < 2; ++j) {
                    Gy_plain[i] += obs_params::G[i][j] * y_test[j];
                }

                xnext_plain[i] = Fx_plain[i] + Gy_plain[i];
            }

            double u_plain = 0.0;
            for (int j = 0; j < 4; ++j) {
                u_plain += obs_params::H[0][j] * xnext_plain[j];
            }

            double u_ckks_test =
                u_terms_msg[0].real()
              + u_terms_msg[1].real()
              + u_terms_msg[2].real()
              + u_terms_msg[3].real();

            std::cout << "\nFx check" << std::endl;
            for (int i = 0; i < 4; ++i) {
                std::cout << "Fx[" << i << "] plain = "
                          << Fx_plain[i]
                          << ", ckks = "
                          << Fx_dec[i]
                          << ", err = "
                          << std::abs(Fx_plain[i] - Fx_dec[i])
                          << std::endl;
            }

            std::cout << "\nGy check" << std::endl;
            for (int i = 0; i < 4; ++i) {
                std::cout << "Gy[" << i << "] plain = "
                          << Gy_plain[i]
                          << ", ckks = "
                          << Gy_dec[i]
                          << ", err = "
                          << std::abs(Gy_plain[i] - Gy_dec[i])
                          << std::endl;
            }

            std::cout << "\nx_next check" << std::endl;
            for (int i = 0; i < 4; ++i) {
                std::cout << "x_next[" << i << "] plain = "
                          << xnext_plain[i]
                          << ", ckks = "
                          << xnext_dec[i]
                          << ", err = "
                          << std::abs(xnext_plain[i] - xnext_dec[i])
                          << std::endl;
            }

            std::cout << "\nHx terms check" << std::endl;
            for (int i = 0; i < 4; ++i) {
                double term_plain = obs_params::H[0][i] * xnext_plain[i];

                std::cout << "term[" << i << "] plain = "
                          << term_plain
                          << ", ckks = "
                          << u_terms_msg[i].real()
                          << ", err = "
                          << std::abs(term_plain - u_terms_msg[i].real())
                          << std::endl;
            }

            std::cout << "\nu check" << std::endl;
            std::cout << "u_plain = " << u_plain << std::endl;
            std::cout << "u_ckks  = " << u_ckks_test << std::endl;
            std::cout << "u_err   = " << std::abs(u_plain - u_ckks_test) << std::endl;

            return 0;
        }

        std::ofstream log_file("ckks_closedloop_200_boot_packed.csv");

        log_file << "k,state_level,boot_count,u_ref,u_ckks,abs_u_err,"
                 << "y0_ref,y0_ckks,y0_err,"
                 << "y1_ref,y1_ckks,y1_err,"
                 << "loop_ms,nonboot_total_ms,client_io_ms,hom_eval_ms,"
                 << "encrypt_y_ms,state_hom_ms,output_hom_ms,decrypt_u_ms,"
                 << "boot_total_ms,boot_pure_ms\n";

        double max_u_err = 0.0;
        int last_valid_step = -1;
        int boot_count = 0;

        double total_loop_ms = 0.0;
        double total_nonboot_total_ms = 0.0;
        double total_client_io_ms = 0.0;
        double total_hom_eval_ms = 0.0;
        double total_encrypt_y_ms = 0.0;
        double total_state_hom_ms = 0.0;
        double total_output_hom_ms = 0.0;
        double total_decrypt_u_ms = 0.0;
        double total_boot_total_ms = 0.0;
        double total_boot_pure_ms = 0.0;

        double max_loop_ms = 0.0;
        double max_nonboot_total_ms = 0.0;
        double max_hom_eval_ms = 0.0;
        double max_boot_total_ms = 0.0;
        double max_boot_pure_ms = 0.0;

        for (int k = 0; k < T; ++k) {
            auto loop_t0 = Clock::now();

            double loop_ms = 0.0;
            double nonboot_total_ms = 0.0;
            double client_io_ms = 0.0;
            double hom_eval_ms = 0.0;
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
            // Plaintext controller driven by the same y_ckks.
            // Expected behavior for the packed CKKS controller.
            // ------------------------------------------------------------
            std::vector<double> xc_ckks_plain_next(4, 0.0);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    xc_ckks_plain_next[i] += obs_params::F[i][j] * xc_ckks_plain[j];
                }

                for (int l = 0; l < 2; ++l) {
                    xc_ckks_plain_next[i] += obs_params::G[i][l] * y_ckks[l];
                }
            }

            xc_ckks_plain = xc_ckks_plain_next;

            u_ckks_plain = 0.0;
            for (int j = 0; j < 4; ++j) {
                u_ckks_plain += obs_params::H[0][j] * xc_ckks_plain[j];
            }

            // ------------------------------------------------------------
            // Bootstrap packed controller state if level is low.
            // One packed state ciphertext only.
            // ------------------------------------------------------------
            if (state_level <= 4) {
                auto boot_total_t0 = Clock::now();

                std::cout << "\nPACKED BOOT trigger at step "
                          << k
                          << ", state_level = "
                          << state_level
                          << std::endl;

                auto level_down_t0 = Clock::now();
                ckks.level_down_to_zero(ct_x);
                double level_down_ms = elapsed_ms(level_down_t0, Clock::now());

                std::vector<Complex> before_boot_slots;
                if (boot_count < 3) {
                    before_boot_slots = ckks.decrypt_vector(ct_x);
                }

                Ct refreshed;

                auto boot_t0 = Clock::now();
                ckks.bootstrap_basic(refreshed, ct_x);
                boot_pure_ms = elapsed_ms(boot_t0, Clock::now());

                if (boot_count < 3) {
                    std::vector<Complex> after_boot_slots =
                        ckks.decrypt_vector(refreshed);

                    std::cout << "  before boot slots[0..7] = ";
                    for (int i = 0; i < 8; ++i) {
                        std::cout << before_boot_slots[i].real();
                        if (i != 7) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << std::endl;

                    std::cout << "  after  boot slots[0..7] = ";
                    for (int i = 0; i < 8; ++i) {
                        std::cout << after_boot_slots[i].real();
                        if (i != 7) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << std::endl;

                    double first4_boot_err = 0.0;
                    for (int i = 0; i < 4; ++i) {
                        first4_boot_err = std::max(
                            first4_boot_err,
                            std::abs(
                                before_boot_slots[i].real()
                              - after_boot_slots[i].real()
                            )
                        );
                    }

                    double tail_abs_after = 0.0;
                    for (int i = 4; i < 8; ++i) {
                        tail_abs_after = std::max(
                            tail_abs_after,
                            std::abs(after_boot_slots[i].real())
                        );
                    }

                    std::cout << "  first4_boot_err = "
                              << first4_boot_err
                              << std::endl;

                    std::cout << "  tail_abs_after slots[4..7] = "
                              << tail_abs_after
                              << std::endl;
                }

                ct_x = std::move(refreshed);

                state_level = ckks.level_of(ct_x);

                boot_count += 1;

                boot_total_ms = elapsed_ms(boot_total_t0, Clock::now());

                total_boot_total_ms += boot_total_ms;
                total_boot_pure_ms += boot_pure_ms;
                max_boot_total_ms = std::max(max_boot_total_ms, boot_total_ms);
                max_boot_pure_ms = std::max(max_boot_pure_ms, boot_pure_ms);

                std::cout << "PACKED BOOT done. New state_level = "
                          << state_level
                          << ", boot_count = "
                          << boot_count
                          << ", level_down_ms = "
                          << level_down_ms
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

            // ------------------------------------------------------------
            // Encrypt current plant output y as one packed ciphertext:
            // [y0, y1, 0, 0, 0, 0, 0, 0, ...]
            // ------------------------------------------------------------
            auto encrypt_y_t0 = Clock::now();

            Ct ct_y;
            ckks.encrypt_periodic4(
                ct_y,
                Vec4{y_ckks[0], y_ckks[1], 0.0, 0.0},
                state_level,
                ct_x.GetScale()
            );

            encrypt_y_ms = elapsed_ms(encrypt_y_t0, Clock::now());

            // ------------------------------------------------------------
            // Packed homomorphic observer state update:
            // ct_x_next = F ct_x + Gbar ct_y
            // ------------------------------------------------------------
            auto state_hom_t0 = Clock::now();

            Ct ct_Fx;
            Ct ct_Gy;
            Ct ct_x_next;

            ckks.packed_matvec4(ct_Fx, F, ct_x);
            ckks.packed_matvec4(ct_Gy, Gbar, ct_y);

            ct_Gy.SetScale(ct_Fx.GetScale());
            ckks.context->Add(ct_x_next, ct_Fx, ct_Gy);

            state_hom_ms = elapsed_ms(state_hom_t0, Clock::now());

            int level_after_state_update = ckks.level_of(ct_x_next);

            // ------------------------------------------------------------
            // Packed Hx terms:
            // ct_u_terms = [h0*x0, h1*x1, h2*x2, h3*x3, ...]
            // Client/actuator decrypts and sums first 4 slots.
            // ------------------------------------------------------------
            auto output_hom_t0 = Clock::now();

            Ct ct_u_terms;
            ckks.packed_h_terms(ct_u_terms, Hmask, ct_x_next);

            output_hom_ms = elapsed_ms(output_hom_t0, Clock::now());

            auto decrypt_u_t0 = Clock::now();
            u_ckks = ckks.decrypt_h_terms_sum4(ct_u_terms);
            decrypt_u_ms = elapsed_ms(decrypt_u_t0, Clock::now());

            double controller_u_err = std::abs(u_ckks_plain - u_ckks);

            if (k < 20 || controller_u_err > 1e-2) {
                Vec4 x_dec = ckks.decrypt_first4(ct_x_next);

                double max_x_controller_err = 0.0;
                for (int i = 0; i < 4; ++i) {
                    max_x_controller_err = std::max(
                        max_x_controller_err,
                        std::abs(xc_ckks_plain[i] - x_dec[i])
                    );
                }

                std::cout << "DEBUG controller k = "
                          << k
                          << ", y_ckks = ["
                          << y_ckks[0]
                          << ", "
                          << y_ckks[1]
                          << "]"
                          << ", u_plain_y = "
                          << u_ckks_plain
                          << ", u_ckks = "
                          << u_ckks
                          << ", controller_u_err = "
                          << controller_u_err
                          << ", max_x_err = "
                          << max_x_controller_err
                          << ", x_plain = ["
                          << xc_ckks_plain[0] << ", "
                          << xc_ckks_plain[1] << ", "
                          << xc_ckks_plain[2] << ", "
                          << xc_ckks_plain[3] << "]"
                          << ", x_dec = ["
                          << x_dec[0] << ", "
                          << x_dec[1] << ", "
                          << x_dec[2] << ", "
                          << x_dec[3] << "]"
                          << std::endl;
            }

            // Update encrypted controller state.
            ct_x = std::move(ct_x_next);
            state_level = level_after_state_update;

            double abs_u_err = std::abs(u_ref - u_ckks);
            double y0_err = std::abs(y_ref[0] - y_ckks[0]);
            double y1_err = std::abs(y_ref[1] - y_ckks[1]);

            max_u_err = std::max(max_u_err, abs_u_err);
            last_valid_step = k;

            hom_eval_ms = state_hom_ms + output_hom_ms;
            client_io_ms = encrypt_y_ms + decrypt_u_ms;
            nonboot_total_ms = client_io_ms + hom_eval_ms;
            loop_ms = elapsed_ms(loop_t0, Clock::now());

            total_loop_ms += loop_ms;
            total_nonboot_total_ms += nonboot_total_ms;
            total_client_io_ms += client_io_ms;
            total_hom_eval_ms += hom_eval_ms;
            total_encrypt_y_ms += encrypt_y_ms;
            total_state_hom_ms += state_hom_ms;
            total_output_hom_ms += output_hom_ms;
            total_decrypt_u_ms += decrypt_u_ms;

            max_loop_ms = std::max(max_loop_ms, loop_ms);
            max_nonboot_total_ms = std::max(max_nonboot_total_ms, nonboot_total_ms);
            max_hom_eval_ms = std::max(max_hom_eval_ms, hom_eval_ms);

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
                     << nonboot_total_ms << ","
                     << client_io_ms << ","
                     << hom_eval_ms << ","
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
                          << ", nonboot_total_ms = " << nonboot_total_ms
                          << ", hom_eval_ms = " << hom_eval_ms
                          << std::endl;
            }
        }

        log_file.close();

        std::cout << "\n=== Packed Summary ===" << std::endl;
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

            std::cout << "Avg non-boot total time [ms]: "
                      << total_nonboot_total_ms / completed_steps
                      << std::endl;
            std::cout << "Max non-boot total time [ms]: "
                      << max_nonboot_total_ms
                      << std::endl;

            std::cout << "Avg client I/O time [ms]: "
                      << total_client_io_ms / completed_steps
                      << std::endl;
            std::cout << "Avg homomorphic evaluation time [ms]: "
                      << total_hom_eval_ms / completed_steps
                      << std::endl;
            std::cout << "Max homomorphic evaluation time [ms]: "
                      << max_hom_eval_ms
                      << std::endl;

            std::cout << "Avg encrypt packed y time [ms]: "
                      << total_encrypt_y_ms / completed_steps
                      << std::endl;
            std::cout << "Avg packed state update time [ms]: "
                      << total_state_hom_ms / completed_steps
                      << std::endl;
            std::cout << "Avg packed Hx terms time [ms]: "
                      << total_output_hom_ms / completed_steps
                      << std::endl;
            std::cout << "Avg decrypt u terms time [ms]: "
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
            std::cout << "Max boot trigger time [ms]: "
                      << max_boot_total_ms
                      << std::endl;
            std::cout << "Max pure Boot() time per refresh [ms]: "
                      << max_boot_pure_ms
                      << std::endl;
        }

        std::cout << "CSV saved to ckks_closedloop_200_boot_packed.csv"
                  << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
