# Cheddar CKKS Observer Controller

Cheddar FHE를 이용하여 다음 observer-based controller를 CKKS 암호문상에서 계산하는 코드이다.

[
x_c(k+1)=F x_c(k)+G y(k),
\qquad
u(k)=H x_c(k)
]

Plant는 다음과 같다.

[
x_p(k+1)=A x_p(k)+B u(k),
\qquad
y(k)=C x_p(k)
]

## Files

### `ckks_sanity.cpp`

Cheddar CKKS의 기본 동작을 확인하는 코드이다.

다음 과정을 테스트한다.

* CKKS encoding
* Encryption
* Ciphertext-constant multiplication
* Rescaling
* Decryption
* Decoding

### `ckks_obs_standalone.cpp`

Observer-controller 한 step을 CKKS로 계산하고 plaintext 결과와 비교한다.

다음 계산을 수행한다.

[
x_c^+=F x_c+G y,
\qquad
u=H x_c^+
]

CKKS 계산 결과와 plaintext 계산 결과가 일치하는지 확인하기 위한 코드이다.

### `ckks_closedloop_200.cpp`

Bootstrapping 없이 CKKS controller와 plant를 연결한 closed-loop simulation이다.

Ciphertext level이 소진될 때까지 실행하며, 현재 parameter에서는 약 32 step까지 계산 가능하다.

### `ckks_closedloop_200_boot.cpp`

Controller state의 각 성분을 별도의 ciphertext로 암호화한 기준 구현이다.

* Controller state ciphertext: 4개
* Plant output ciphertext: 2개
* 200-step closed-loop simulation
* 반복 bootstrapping 적용
* Correctness 검증 완료

State ciphertext 4개를 각각 bootstrapping하므로 refresh 한 번에 약 6초가 소요된다.

### `ckks_closedloop_200_boot_packed.cpp`

Controller state를 하나의 CKKS ciphertext에 packing한 구현이다.

State packing:

```text
[x0, x1, x2, x3, 0, 0, ...]
```

Plant output packing:

```text
[y0, y1, 0, 0, 0, 0, ...]
```

Packed matrix-vector multiplication은 diagonal method를 사용한다.

[
Fx=\sum_d D_d(F)\operatorname{Rot}(x,d)
]

(G\in\mathbb{R}^{4\times2})는 다음과 같이 (4\times4) 행렬로 zero-padding한 뒤 같은 방법을 적용한다.

[
\bar G=
\begin{bmatrix}
G_{00}&G_{01}&0&0\
G_{10}&G_{11}&0&0\
G_{20}&G_{21}&0&0\
G_{30}&G_{31}&0&0
\end{bmatrix}
]

Output 계산에서는 먼저 다음 encrypted slot-wise product를 계산한다.

```text
[h0*x0, h1*x1, h2*x2, h3*x3, 0, ...]
```

그 후 client가 ciphertext를 복호화하고 처음 네 slot을 더하여 (u)를 계산한다.

현재 검증 결과:

* Simulation steps: 200
* Bootstrapping count: 16
* Initial encryption level: 19
* Bootstrapping trigger level: 4
* Maximum control-input error: `0.00278943`
* Average pure bootstrapping time: `1446.18 ms`
* Tested GPU: NVIDIA RTX 4060

Packing을 적용하여 controller state refresh에 필요한 bootstrapping 횟수가 4회에서 1회로 감소하였다.

## Parameter files

현재 simulation에서는 Cheddar의 다음 parameter file을 사용한다.

```text
~/Cheddar_project/cheddar-fhe/parameters/bootparam_35.json
```

주요 parameter는 다음과 같다.

* Polynomial degree: (N=2^{16}=65536)
* Number of complex slots: 32768
* Default scale: approximately (2^{35})
* Default encryption level: 19
* Maximum level: 31
* `num_cts_levels`: 4
* `num_stc_levels`: 3

Controller ciphertext는 반드시 `default_encryption_level`에서 생성해야 한다.

```cpp
int state_level = ckks.default_encryption_level;
```

`param->max_level_`에서 시작하면 bootstrapping 입력 scale이 맞지 않아 refresh 결과가 손상될 수 있다.

## Dependencies

* NVIDIA CUDA-capable GPU
* CUDA Toolkit 12.4
* CMake 3.24 or newer
* `nlohmann-json3-dev`
* Cheddar FHE

Cheddar repository는 다음 위치에 있다고 가정한다.

```text
~/Cheddar_project/cheddar-fhe
```

필요한 package 설치:

```bash
sudo apt update
sudo apt install nlohmann-json3-dev
```

## Build

현재 디렉터리에서 다음 명령을 실행한다.

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.4

cmake --build build -j
```

## Run

기본 CKKS 테스트:

```bash
./build/ckks_sanity
```

Observer 한 step 테스트:

```bash
./build/ckks_obs_standalone
```

Bootstrapping 없는 closed-loop simulation:

```bash
./build/ckks_closedloop_200
```

Scalar ciphertext bootstrapping simulation:

```bash
./build/ckks_closedloop_200_boot
```

Packed ciphertext self-test:

```bash
PACKED_SELF_TEST=1 \
./build/ckks_closedloop_200_boot_packed
```

Packed 200-step closed-loop simulation:

```bash
./build/ckks_closedloop_200_boot_packed
```

실행 결과는 다음 CSV 파일로 저장된다.

```text
ckks_closedloop_200_boot_packed.csv
```

## GPU memory note

RTX 4060 8 GB 환경에서는 `bootparam_35.json`을 바로 초기화할 때 GPU memory allocation이 실패할 수 있다.

현재 코드는 같은 process에서 `bootparam_30.json` 초기화를 먼저 시도하고 예상되는 out-of-memory exception을 처리한 뒤 `bootparam_35.json`을 초기화한다.

따라서 실행 중 다음 메시지가 나타나는 것은 현재 환경에서는 예상된 동작이다.

```text
Dummy warm-up failed/caught as expected:
std::bad_alloc: out_of_memory
```

## Acknowledgment and Citation

This implementation uses
[Cheddar](https://github.com/scale-snu/cheddar-fhe),
a GPU-accelerated CKKS fully homomorphic encryption library.

The CKKS and bootstrapping parameters used in this project are based on
Cheddar's `bootparam_35.json`.

When using this repository, please also cite the Cheddar paper:

```bibtex
@inproceedings{asplos-2026-cheddar,
  author = {Choi, Wonseok and Kim, Jongmin and Ahn, {Jung Ho}},
  title = {Cheddar: {A} Swift Fully Homomorphic Encryption Library Designed for {GPU} Architectures},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
  year = {2025},
  url = {https://doi.org/10.1145/3760250.3762223},
  doi = {10.1145/3760250.3762223}
}