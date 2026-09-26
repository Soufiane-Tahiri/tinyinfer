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
| ESP32 | **520KB SRAM** |  Primary — everything is designed around this constraint |
| Raspberry Pi 3B+ | ~512MB |  Secondary validation — numerical comparison against PyTorch |
| x86 Linux | unlimited |  Development host |

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
- **Test accuracy:** ~0.88, recall ~0.86 (PyTorch reference, seed 42, 5% FPR budget threshold from grouped OOF). C-side validation pending loader completion.
- **Training + export:** `tiny_mlp/nsl-kdd-tinyinfer.ipynb` → `tiny_mlp/tinyinfer_weights.npz` + `tiny_mlp/tinyinfer_meta.json` → `tools/export.py` → `model_f32.bin` / `model_int8.bin`

**Honest limitation:** NSL-KDD's features are windowed traffic/host statistics computed over a connection, not something an ESP32 can extract from raw packets in real time. This demo proves the inference engine — correct output on pre-extracted feature vectors, on-device, in float32 and int8 — not a complete on-device intrusion detection pipeline. Feature extraction from live traffic is out of scope for now.

---

## Architecture

```
tinyinfer/
├── include/          # Public headers
├── src/              # Implementation (tensor, layers, activations)
├── tests/            # Numerical validation against PyTorch reference outputs
├── tools/
│   └── export.py     # PyTorch → tinyinfer binary weight exporter (float32 + int8)
├── tiny_mlp/          # NSL-KDD demo model: notebook, weights, metadata
├── main.c
├── CMakeLists.txt
└── README.md
```

---

## Core design decisions

### Memory model
- Static allocation — no heap in the inference path, ever
- Hard SRAM budget: fits inside ESP32's 520KB with room for FreeRTOS overhead
- Planned, not yet implemented: compile-time weight buffer sizing via static
  assertions, flash (SPIFFS) weight storage, ping-pong activation buffer reuse

### Tensor representation
- `float32` and `int8` supported — dtype is a first-class field
- Stride-based indexing
- No autograd — inference only

### Weight format
- Custom binary, little-endian throughout: magic bytes + version + dtype tag + layer count + input/output dims
- **float32:** per layer — input/output dims, raw float32 weights, raw float32 bias
- **int8:** per layer — input/output dims, a per-layer `requant_scale`, int8 weights, int32 bias (pre-quantized into the same integer domain as the weight/input product, so the kernel's `acc * requant_scale` lands directly in the next layer's int8 range)
- `tools/export.py` converts a trained PyTorch model to both formats and self-checks each against saved reference outputs before writing them
- A companion `model_meta.json` carries everything that isn't weights: feature order, preprocessing (log1p columns, protocol one-hot, standardization mean/scale, clip bound), and the decision threshold

---

## Roadmap

### v0.1 — Host-validated MLP
- [x] `Tensor` struct — aligned allocation, overflow-safe shape computation
- [x] Linear layer forward pass (float32 + int8, fused ReLU)
- [x] ReLU, Softmax (float32 + int8)
- [x] INT8 requantization in dense kernel (`requant_scale`, `lrintf` rounding)
- [x] Overflow guard on shape parsing, layer dimension bounds checks
- [x] export.py — writes float32 and int8 binary formats, self-checks against reference outputs
- [x] NSL-KDD training notebook — grouped CV, FPR-calibrated threshold, int8 simulation
- [ ] Binary weight loader (C side — in progress)
- [ ] main.c demo — load model, run forward pass, print result and latency
- [ ] End-to-end numerical validation: C output matches PyTorch reference within tolerance

### v0.2 — ESP32 port
- [ ] ESP-IDF port — fit MLP in 520KB SRAM
- [ ] UART output for inference result logging
- [ ] Example: real-time anomaly detection on ESP32
- [ ] SPIFFS weight loading from flash

### v0.3 — Validation + expansion
- [ ] Raspberry Pi validation — compare output vs PyTorch numerically
- [ ] Sigmoid, Tanh
- [ ] Conv2D
- [ ] Ping-pong activation buffer reuse across layers

---

## Build

```bash
# Development host
make

# ESP32 via ESP-IDF
idf.py build
idf.py flash monitor
```

---

## Requirements

- C99 compiler (gcc or clang)
- ESP-IDF v5.x for ESP32 target
- Python 3.x + PyTorch for weight export only

---

## Author

**Soufiane Tahiri**
Master's student — Intelligence et Sécurité des Objets Connectés
Université Moulay Ismail, Faculté des Sciences — Meknès

> If it doesn't fit in 520KB, the architecture is wrong.