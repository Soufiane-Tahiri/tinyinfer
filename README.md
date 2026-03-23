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

## Architecture

```
tinyinfer/
├── include/          # Public headers
├── src/              # Implementation
├── tests/            # Numerical validation against PyTorch reference outputs
├── tools/
│   └── export.py     # PyTorch → tinyinfer binary weight exporter
├── main.c
├── CMakeLists.txt
└── README.md
```

---

## Core design decisions

### Memory model
- Static allocation — no heap in the inference path, ever
- All weight buffers sized at compile time, enforced via static assertions
- Weights stored in flash (SPIFFS), loaded to SRAM per layer
- Activation buffers reused across layers via ping-pong strategy
- Hard SRAM budget: fits inside ESP32's 520KB with room for FreeRTOS overhead

### Tensor representation
- `float32` and `int8` supported — dtype is a first-class field
- Stride-based indexing
- No autograd — inference only

### Weight format
- Custom binary: magic bytes + layer count + per-layer shape + raw float32 data
- `tools/export.py` converts any PyTorch model to this format
- Little-endian throughout — matches ESP32 and x86

---

## Roadmap

### v0.1 — Host-validated MLP
- [ ] `Tensor` struct — static allocation, stride indexing
- [ ] Linear layer forward pass
- [ ] ReLU, Softmax
- [ ] Binary weight loader
- [ ] PyTorch export script
- [ ] Numerical validation test suite

### v0.2 — ESP32 port
- [ ] ESP-IDF port — fit MLP in 520KB SRAM
- [ ] UART output for inference result logging
- [ ] Example: real-time anomaly detection on ESP32

### v0.3 — Validation + expansion
- [ ] Raspberry Pi validation — compare output vs PyTorch numerically
- [ ] Sigmoid, Tanh
- [ ] Conv2D
- [ ] INT8 quantization

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