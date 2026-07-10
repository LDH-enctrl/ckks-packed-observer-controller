#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

using json = nlohmann::json;

using Word = std::uint32_t;
using Complex = cheddar::Complex;

using Ct = cheddar::Ciphertext<Word>;
using Pt = cheddar::Plaintext<Word>;
using Param = cheddar::Parameter<Word>;
using Ctx = cheddar::Context<Word>;
using CtxPtr = cheddar::ContextPtr<Word>;
using UI = cheddar::UserInterface<Word>;

using BootCtx = cheddar::BootContext<Word>;
using BootCtxPtr = std::shared_ptr<BootCtx>;
using EvkRequest = cheddar::EvkRequest;

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

        // Match official Bootstrapping.cpp:
        // boot_context->AddRequiredRotations(req, num_slots);
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

    void encode_vector(Pt& pt, const std::vector<Complex>& msg, int level) {
        double scale = scale_at_level(level);
        context->encoder_.Encode(pt, level, scale, msg);
    }

    void encrypt_vector(Ct& ct, const std::vector<Complex>& msg, int level) {
        Pt pt;
        encode_vector(pt, msg, level);
        interface->Encrypt(ct, pt);
    }

    void decrypt_vector(std::vector<Complex>& msg, const Ct& ct) {
        Pt pt;
        interface->Decrypt(pt, ct);
        context->encoder_.Decode(msg, pt);
    }

    void bootstrap_basic(Ct& out, const Ct& in) {
        boot_context->Boot(out, in, interface->GetEvkMap());
    }
};

static void clear_cuda_error_state() {
    cudaGetLastError();
}

static std::string param_path(const std::string& name) {
    return std::string(std::getenv("HOME")) +
           "/Cheddar_project/cheddar-fhe/parameters/" +
           name;
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

    std::cout << "Continue to actual boot sanity." << std::endl;
}

static int run_actual_boot_sanity() {
    std::cout << "\n=== Actual boot sanity with bootparam_35.json ==="
              << std::endl;

    CheddarBoot ckks(param_path("bootparam_35.json"));

    const int num_slots = ckks.boot_num_slots;

    std::vector<Complex> expected(num_slots);

    for (int i = 0; i < num_slots; ++i) {
        double real = 0.123 + 1e-6 * static_cast<double>(i % 100);
        double imag = -0.045 + 1e-6 * static_cast<double>((i * 7) % 100);
        expected[i] = Complex(real, imag);
    }

    Ct ct;
    Ct ct_boot;

    // Match official Bootstrapping.cpp:
    // EncodeAndEncrypt(ct1, msg1, 0)
    ckks.encrypt_vector(ct, expected, 0);

    std::cout << "Calling Boot-Basic..." << std::endl;

    ckks.bootstrap_basic(ct_boot, ct);

    std::vector<Complex> obtained;
    ckks.decrypt_vector(obtained, ct_boot);

    double max_err = 0.0;
    double avg_err = 0.0;

    int compare_size = std::min(
        static_cast<int>(expected.size()),
        static_cast<int>(obtained.size())
    );

    for (int i = 0; i < compare_size; ++i) {
        double dr = expected[i].real() - obtained[i].real();
        double di = expected[i].imag() - obtained[i].imag();
        double err = std::sqrt(dr * dr + di * di);

        max_err = std::max(max_err, err);
        avg_err += err;
    }

    avg_err /= static_cast<double>(compare_size);

    std::cout << "\n=== Boot sanity result ===" << std::endl;
    std::cout << "compare_size = " << compare_size << std::endl;
    std::cout << "expected[0] = " << expected[0] << std::endl;
    std::cout << "obtained[0] = " << obtained[0] << std::endl;
    std::cout << "max_err = " << max_err << std::endl;
    std::cout << "avg_err = " << avg_err << std::endl;

    return 0;
}

int main() {
    try {
        try_dummy_boot_warmup();
        return run_actual_boot_sanity();
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
