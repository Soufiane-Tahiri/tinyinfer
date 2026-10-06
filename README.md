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

## Why C, why from scratch

- PyTorch on 520KB SRAM is impossible. TFLite exists but understanding it requires knowing what it does internally — this project builds that understanding from zero.
- Every byte is explicit. No GC, no hidden allocations, no runtime surprises.
- **IoT security angle:** inference runs entirely on-device. No cloud dependency, no data leaving the node, no network attack surface. That's an architectural security property, not a feature.

---

## Demo model: network intrusion classifier (NSL-KDD)

To exercise the engine end-to-end, tinyinfer runs a small MLP trained on [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), classifying network connections as normal or attack.

- **Architecture:** 40 → 32 → 16 → 2, ~1.9K parameters (7.4 KB float32, 2.0 KB int8)
- **Test accuracy:** ~0.88, recall ~0.86 (PyTorch reference, seed 42, 5% FPR budget threshold from grouped OOF). C-side validation pending end-to-end run.
- **Training + export:** `tiny_mlp/nsl-kdd-tinyinfer.ipynb` → `tiny_mlp/tinyinfer_weights.npz` + `tiny_mlp/tinyinfer_meta.json` → `tools/export.py` → `model_f32.bin` / `model_int8.bin`

**Honest limitation:** NSL-KDD features are windowed traffic/host statistics computed over a connection, not something an ESP32 can extract from raw packets in real time. This demo proves the inference engine — correct output on pre-extracted feature vectors, on-device, in float32 and int8 — not a complete on-device intrusion detection pipeline. Feature extraction from live traffic is out of scope for now.

---

## Architecture

```
tinyinfer/
├── include/
│   ├── activations.h
│   ├── inference.h
│   ├── layers.h
│   ├── model.h
│   ├── tensor.h
│   └── tinyinfer.h
├── models/
│   ├── model_f32.bin
│   ├── model_int8.bin
│   └── model_meta.json
├── src/
│   ├── activations.c
│   ├── layers.c
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

**Host engine** (`src/`) — used on x86 Linux and Raspberry Pi for development and validation.
Supports batched inference, dynamic tensor shapes, and extensible layer types. Used to validate
numerical correctness against PyTorch before targeting hardware.

**ESP32 runtime** (`tinyinfer.c`) — the bare-metal production kernel.
Loads the entire model binary into one contiguous buffer, points directly into it for weights and
biases (zero per-layer allocation), processes one sample at a time. Compiles cleanly under ESP-IDF
with no dependency on the host engine.

Both engines consume the same binary format produced by `export.py`. For a given input, their
outputs must match within float32 precision — that agreement is the validation proof.

---

## Core design decisions

### Memory model
- Single contiguous buffer load: entire model file read into RAM once, weight/bias pointers set directly into it — no per-layer allocation
- Hard SRAM budget: model + activations + FreeRTOS overhead fits inside ESP32's 520KB
- Planned for v0.2: static arena replacing `malloc`, SPIFFS weight loading from flash

### Weight format
Custom binary, little-endian throughout:
```
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
  [3B] padding
  --- int8 only ---
  [4B] requant_scale (float32)
  ---
  [out_dim × in_dim × elem_size] weights (row-major)
  [out_dim × bias_elem_size]     bias
```

`export.py` writes both formats and self-checks against saved reference logits before reporting success.

---

## Roadmap

### v0.1 — Host-validated MLP
- [x] Tensor struct — aligned allocation, overflow-safe shape computation
- [x] Dense layer forward pass (float32 + int8, fused ReLU, tiled + unrolled)
- [x] INT8 requantization in dense kernel (`requant_scale`, `lrintf` rounding)
- [x] Overflow guard on shape parsing, layer dimension bounds checks
- [x] ReLU activation (float32 + int8)
- [x] export.py — float32 + int8 binary formats, self-check against reference outputs
- [x] NSL-KDD training notebook — grouped CV, FPR-calibrated threshold, int8 simulation
- [x] ESP32 runtime (`tinyinfer.c`) — single-buffer loader, float32 + int8 inference kernels
- [ ] main.c — load model, run forward pass, compare against PyTorch reference, print result + latency
- [ ] End-to-end numerical validation: C output matches PyTorch reference within float32 tolerance

### v0.2 — ESP32 port
- [ ] ESP-IDF port — compile and flash under ESP-IDF v5.x
- [ ] UART output for inference result and latency logging
- [ ] SPIFFS weight loading from flash
- [ ] Static arena replacing `malloc` for zero heap fragmentation

### v0.3 — Expansion
- [ ] Raspberry Pi numerical validation vs PyTorch
- [ ] Latency benchmark vs TFLite Micro on same model and hardware
- [ ] Conv2D layer (host engine first, then ESP32 runtime)
- [ ] Sigmoid, Tanh

---

## Build

```bash
# Development host (x86 Linux)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
./tinyinfer

# ESP32 via ESP-IDF
idf.py build
idf.py flash monitor
```

---

## Requirements

- C99 compiler (gcc or clang)
- CMake 3.16+
- ESP-IDF v5.x for ESP32 target
- Python 3.x + NumPy + PyTorch for weight export only

---

## Author

**Soufiane Tahiri**
Master's student — Intelligence et Sécurité des Objets Connectés
Université Moulay Ismail, Faculté des Sciences — Meknès

> If it doesn't fit in 520KB, the architecture is wrong.