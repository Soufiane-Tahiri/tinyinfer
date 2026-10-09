# tinyinfer

> A from-scratch neural network inference engine in pure C, targeting embedded systems.
> Train in Python. Export weights. Run inference on an ESP32.

---

## What this is

`tinyinfer` is a minimal, dependency-free neural network inference engine written in C99.

No PyTorch. No TensorFlow Lite. No abstraction layers hiding what's actually happening at the memory level.

The contract is simple: **train a model in Python, export the weights, run inference in C on bare metal.**

Primary target is an **ESP32 with 520KB SRAM**. If it fits and runs correctly there, it runs anywhere.

---

## Target hardware

| Platform | RAM | Role |
|---|---|---|
| ESP32 | **520KB SRAM** | Primary — everything is designed around this constraint |
| Raspberry Pi 3B+ | ~512MB | Secondary validation — numerical comparison against PyTorch |
| x86 Linux | unlimited | Development host |

ESP32 first. Always. The Pi is for validation, not for setting the bar.

---

## Bare-Metal Performance (ESP32)

Compiled via ESP-IDF with `-O3` optimization for the Xtensa LX6 (240MHz).

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Execution Latency** | 324 µs | Single INT8 forward pass. |
| **Max Throughput** | 3,086 inf/sec | |
| **Static RAM (BSS/Data)**| 13.2 KB (4%) | Includes FreeRTOS & ESP-IDF. |
| **Dynamic RAM (Heap)** | 0 bytes | Bypasses `malloc()` completely during runtime. |
| **Flash Usage** | 180.9 KB (18%) | Model binary mapped directly from `.rodata`. |

---

## Why C, why from scratch

- PyTorch on 520KB SRAM is impossible. TFLite exists but understanding it requires knowing what it does internally — this project builds that understanding from zero.
- Every byte is explicit. No GC, no hidden allocations, no runtime surprises.
- **IoT security angle:** inference runs entirely on-device. No cloud dependency, no data leaving the node, no network attack surface. That's an architectural security property, not a feature.

---

## Demo model: network intrusion classifier (NSL-KDD)

To exercise the engine end-to-end, tinyinfer runs a small MLP trained on [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), classifying network connections as normal or attack.

- **Architecture:** 40 → 32 → 16 → 2, ~1.9K parameters (7.4 KB float32, 2.0 KB int8)
- **Test accuracy:** ~0.88, recall ~0.86 (PyTorch reference, seed 42, 5% FPR budget threshold from grouped OOF)
- **Host validation:** all four inference paths (framework f32, framework int8, kernel f32, kernel int8) pass against PyTorch reference logits within float32 tolerance
- **Training + export:** `tiny_mlp/nsl-kdd-tinyinfer.ipynb` → `tiny_mlp/tinyinfer_weights.npz` + `tiny_mlp/tinyinfer_meta.json` → `tools/export.py` → `models/model_f32.bin` / `models/model_int8.bin`

**Honest limitation:** NSL-KDD features are windowed traffic/host statistics computed over a connection, not something an ESP32 can extract from raw packets in real time. This demo proves the inference engine — correct output on pre-extracted feature vectors, on-device, in float32 and int8 — not a complete on-device intrusion detection pipeline. Feature extraction from live traffic is the focus of v0.3.

---

## Architecture

```text
tinyinfer/
├── include/
│   ├── activations.h
│   ├── inference.h
│   ├── layers.h
│   ├── model.h
│   ├── tensor.h
│   ├── tinyinfer.h
│   └── model_data.h
├── models/
│   ├── model_f32.bin
│   ├── model_int8.bin
│   └── model_meta.json
├── src/
│   ├── activations.c
│   ├── layers.c
│   ├── main.c
│   ├── model.c
│   ├── tensor.c
│   └── tinyinfer.c
├── tests/
│   ├── test_activations.c
│   ├── test_layer.c
│   ├── test_layer_int8.c
│   └── test_tensor.c
├── tiny_mlp/
│   ├── nsl-kdd-tinyinfer.ipynb
│   ├── tinyinfer_meta.json
│   ├── tinyinfer_mlp.pth
│   └── tinyinfer_weights.npz
├── tools/
│   └── export.py
├── .gitignore
├── CMakeLists.txt
└── README.md

```

### Two-engine design

**Host engine** (`src/model.c`) — used on x86 Linux and Raspberry Pi for development and validation.
Supports batched inference, dynamic tensor shapes, and extensible layer types. Used to validate
numerical correctness against PyTorch before targeting hardware.

**ESP32 runtime** (`src/tinyinfer.c`) — the bare-metal production kernel.
Loads the entire model binary into one contiguous buffer, points directly into it for weights and
biases (zero per-layer allocation), and processes one sample at a time on the stack. Compiles cleanly under ESP-IDF with no dependency on the host engine.

Both engines consume the same binary format produced by `export.py`. For a given input, their outputs must match within float32 precision — that agreement is the validation proof.

---

## Core design decisions

### Memory model

* **Zero-Copy Flash Loader:** Model weights are embedded as a `const uint8_t[]` C-array baked into the firmware binary via `xxd -i`. The ESP32 MMU maps this directly into XIP (execute-in-place) flash. Zero file I/O, zero filesystem overhead.
* **Zero Heap Allocation:** Bypassed `malloc` completely during the execution phase. Intermediate activation buffers are statically allocated on FreeRTOS stack frames, resulting in 0 bytes of dynamic heap usage and zero heap fragmentation.

### Weight format

Custom binary, little-endian throughout:

```text
Header (24 bytes):
  [4B] magic "TINF"
  [4B] version (1)
  [4B] dtype (0=float32, 1=int8)
  [4B] num_layers
  [4B] in_dim
  [4B] out_dim

int8 header extension:
  [4B] input_scale (float32)

Per layer:
  [4B] in_dim
  [4B] out_dim
  [1B] activation (0=none, 1=relu)
  [3B] padding (alignment)
  --- int8 only ---
  [4B] requant_scale (float32)
  ---
  [out_dim × in_dim × elem_size] weights (row-major)
  [out_dim × bias_elem_size]     bias

```

`export.py` writes both formats and self-checks against saved reference logits before reporting success.
Int8 dequantization: `float_logit = raw_int32 × requant_scale_final × int8_output_scale`

---

## Roadmap

### v0.1 — Host-validated MLP ✓

* [x] Tensor struct — aligned allocation, overflow-safe shape computation, TI_INT32 support
* [x] Dense layer forward pass (float32 + int8, fused ReLU, tiled + unrolled)
* [x] INT8 requantization in dense kernel (`requant_scale`, `lrintf` rounding)
* [x] Overflow guard on shape parsing, layer dimension bounds checks
* [x] ReLU activation (float32 + int8)
* [x] Model graph abstraction — binary loader, ping-pong activation buffers
* [x] ESP32 runtime (`tinyinfer.c`) — single-buffer loader, float32 + int8 kernels
* [x] Argmax + margin decision utilities (`inference.h`, static inline)
* [x] export.py — float32 + int8 binary formats, activation byte per layer, self-check
* [x] NSL-KDD training notebook — grouped CV, FPR-calibrated threshold, int8 simulation
* [x] main.c — four-path validation harness, ALL PASS against PyTorch reference

### v0.2 — ESP32 Bare Metal Port ✓

* [x] ESP-IDF port — compile and flash under ESP-IDF v5.x
* [x] Model weights as `const uint8_t[]` C-array baked into firmware
* [x] `ti_load_model_int8_mem()` — direct pointer mapping to `.rodata`, no file I/O
* [x] UART hardware profiling via `esp_timer_get_time()`
* [x] Zero dynamic heap allocation (`esp_get_free_heap_size()` verified)

### v0.3 — Live Data Ingestion (In Progress)

* [ ] Promiscuous WiFi packet sniffing on ESP32
* [ ] On-device NSL-KDD feature extraction and connection state tracking
* [ ] Real-time local scaling (Min-Max/Z-Score) in C
* [ ] Raspberry Pi numerical validation vs PyTorch (carryover)
* [ ] Latency benchmark vs TFLite Micro on same model and hardware

---

## Build

```bash
# Development host (x86 Linux / Windows)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
./tinyinfer

# Generate C-array from model binary (for ESP32 firmware)
xxd -i models/model_int8.bin > include/model_data.h

# ESP32 via ESP-IDF (PlatformIO)
pio run -t upload -t monitor

```

---

## Requirements

* C99 compiler (gcc or clang)
* CMake 3.16+
* ESP-IDF v5.x / PlatformIO for ESP32 target
* Python 3.x + NumPy + PyTorch for weight export only

---

## Author

**Soufiane Tahiri**
Master's student — Intelligence et Sécurité des Objets Connectés
Université Moulay Ismail, Faculté des Sciences — Meknès

> If it doesn't fit in 520KB, the architecture is wrong.