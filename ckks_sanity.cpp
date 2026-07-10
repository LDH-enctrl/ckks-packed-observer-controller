#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <memory>
#include <string>
#include <cstdint>

#include <nlohmann/json.hpp>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Parameter.h"


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

int main() {
    try {
        std::string param_path =
            std::string(std::getenv("HOME")) +
            "/Cheddar_project/cheddar-fhe/parameters/bootparam_30.json";

        CheddarSimple ckks(param_path);

        int level = ckks.param->max_level_;

        Ct ct_x;
        ckks.encrypt_scalar(ct_x, 0.123, level);

        Ct ct_y;
        ckks.multiply_const_rescale(ct_y, ct_x, 2.0, level);

        double y = ckks.decrypt_scalar(ct_y);

        std::cout << "expected: " << 0.246 << std::endl;
        std::cout << "obtained: " << y << std::endl;
        std::cout << "abs error: " << std::abs(y - 0.246) << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}