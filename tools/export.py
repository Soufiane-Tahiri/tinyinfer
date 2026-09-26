
import json
import os
import struct
import numpy as np

MAGIC = b"TINF"
VERSION = 1
DTYPE_F32 = 0
DTYPE_INT8 = 1

WEIGHTS_NPZ = "../tiny_mlp/tinyinfer_weights.npz"
META_JSON = "../tiny_mlp/tinyinfer_meta.json"
OUT_F32_BIN = "model_f32.bin"
OUT_INT8_BIN = "model_int8.bin"
OUT_META = "model_meta.json"

E = "<"


def load_sources():
    import sys
    npz = np.load(WEIGHTS_NPZ)
    meta = json.load(open(META_JSON))

    arch_dims = [int(x) for x in meta["architecture"].split("-")]
    n_layers = len(arch_dims) - 1

    n_layers_npz = sum(1 for k in npz.files if k.startswith("W") and not k.endswith("_q"))
    if n_layers_npz != n_layers:
        print(f"ERROR: layer count mismatch — npz has {n_layers_npz} weight arrays, "
              f"meta architecture '{meta['architecture']}' implies {n_layers} layers. "
              f"npz and meta.json are probably from different training runs.")
        sys.exit(1)

    for i in range(n_layers):
        expected = (arch_dims[i + 1], arch_dims[i])
        actual = tuple(npz[f"W{i}"].shape)
        if actual != expected:
            print(f"ERROR: W{i} shape {actual} != expected {expected} "
                  f"from architecture '{meta['architecture']}'")
            sys.exit(1)

    return npz, meta, arch_dims, n_layers

def compute_int8_layer_params(npz, meta, n_layers):
    in_scale = float(npz["in_scale"][0])
    w_scales = [float(s) for s in npz["w_scales"]]
    act_scales = [float(s) for s in npz["act_scales"]]
    assert len(w_scales) == n_layers and len(act_scales) == n_layers

    layer_input_scale = [in_scale] + act_scales[:-1]

    quantized_bias = []
    requant_scale = []
    for i in range(n_layers):
        b = npz[f"b{i}"].astype(np.float64)
        combined = layer_input_scale[i] * w_scales[i]
        bq = np.round(b / combined).astype(np.int32)
        quantized_bias.append(bq)
        requant_scale.append(combined / act_scales[i])

    return in_scale, w_scales, act_scales, quantized_bias, requant_scale

def write_model_f32(npz, arch_dims, n_layers, path):
    with open(path, "wb") as f:
        f.write(struct.pack(E + "4s", MAGIC))
        f.write(struct.pack(E + "iiiii", VERSION, DTYPE_F32, n_layers,
                            arch_dims[0], arch_dims[-1]))
        for i in range(n_layers):
            in_dim, out_dim = arch_dims[i], arch_dims[i + 1]
            f.write(struct.pack(E + "ii", in_dim, out_dim))
            W = npz[f"W{i}"].astype("<f4")
            b = npz[f"b{i}"].astype("<f4")
            assert W.shape == (out_dim, in_dim) and b.shape == (out_dim,)
            f.write(W.tobytes(order="C"))
            f.write(b.tobytes(order="C"))


def write_model_int8(npz, meta, arch_dims, n_layers, path):
    in_scale, w_scales, act_scales, quantized_bias, requant_scale = \
        compute_int8_layer_params(npz, meta, n_layers)

    with open(path, "wb") as f:
        f.write(struct.pack(E + "4s", MAGIC))
        f.write(struct.pack(E + "iiiii", VERSION, DTYPE_INT8, n_layers,
                            arch_dims[0], arch_dims[-1]))
        f.write(struct.pack(E + "f", in_scale))

        for i in range(n_layers):
            in_dim, out_dim = arch_dims[i], arch_dims[i + 1]
            f.write(struct.pack(E + "ii", in_dim, out_dim))
            f.write(struct.pack(E + "f", requant_scale[i]))

            Wq = npz[f"W{i}_q"].astype("<i1")
            bq = quantized_bias[i].astype("<i4")
            assert Wq.shape == (out_dim, in_dim) and bq.shape == (out_dim,)
            f.write(Wq.tobytes(order="C"))
            f.write(bq.tobytes(order="C"))

    return act_scales


def write_shared_meta(meta, act_scales, path):
    sp = meta["scaler_params"]
    assert "clip_std" in sp, "scaler_params missing clip_std — regenerate tinyinfer_meta.json"
    out = {
        "architecture": meta["architecture"],
        "logit_diff_threshold": meta["logit_diff_threshold"],
        "clip_std": sp["clip_std"],
        "feature_names": sp["feature_names"],
        "log1p_cols": sp["log1p_cols"],
        "protocols": sp["protocols"],
        "mean_": sp["mean_"],
        "scale_": sp["scale_"],
        "int8_output_scale": act_scales[-1],
    }
    json.dump(out, open(path, "w"), indent=2)


def read_model_f32(path):
    with open(path, "rb") as f:
        head_fmt = E + "4siiiii"
        magic, version, dtype_tag, n_layers, input_dim, output_dim = struct.unpack(
            head_fmt, f.read(struct.calcsize(head_fmt))
        )
        assert magic == MAGIC and dtype_tag == DTYPE_F32
        layers = []
        for _ in range(n_layers):
            in_dim, out_dim = struct.unpack(E + "ii", f.read(8))
            W = np.frombuffer(f.read(out_dim * in_dim * 4), dtype="<f4").reshape(out_dim, in_dim)
            b = np.frombuffer(f.read(out_dim * 4), dtype="<f4")
            layers.append((W, b))
    return layers


def read_model_int8(path):
    with open(path, "rb") as f:
        head_fmt = E + "4siiiii"
        magic, version, dtype_tag, n_layers, input_dim, output_dim = struct.unpack(
            head_fmt, f.read(struct.calcsize(head_fmt))
        )
        assert magic == MAGIC and dtype_tag == DTYPE_INT8
        in_scale = struct.unpack(E + "f", f.read(4))[0]
        layers = []
        for _ in range(n_layers):
            in_dim, out_dim = struct.unpack(E + "ii", f.read(8))
            rscale = struct.unpack(E + "f", f.read(4))[0]
            W = np.frombuffer(f.read(out_dim * in_dim), dtype="<i1").reshape(out_dim, in_dim)
            b = np.frombuffer(f.read(out_dim * 4), dtype="<i4")
            layers.append((W, b, rscale))
    return in_scale, layers


def forward_f32(layers, x):
    h = x.astype(np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def requant(acc, rscale, relu):
    q = np.round(acc.astype(np.float64) * rscale)
    if relu:
        return np.clip(q, 0, 127).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int32)


def quantize_input(x, in_scale):
    return np.clip(np.round(x / in_scale), -128, 127).astype(np.int32)


def forward_int8(in_scale, layers, x):
    h = quantize_input(x, in_scale)
    for i, (W, b, rscale) in enumerate(layers):
        acc = h.astype(np.int64) @ W.astype(np.int64).T + b.astype(np.int64)
        relu = i < len(layers) - 1
        h = requant(acc, rscale, relu)
    return h


def self_check_f32(npz, path):
    layers = read_model_f32(path)
    logits = forward_f32(layers, npz["ref_inputs"])
    err = np.abs(logits - npz["ref_logits"]).max()
    return err < 1e-3, err


def self_check_int8(npz, path, act_scale_out, min_agreement=0.97):
    in_scale, layers = read_model_int8(path)
    out_int8 = forward_int8(in_scale, layers, npz["ref_inputs"])
    dequant = out_int8.astype(np.float64) * act_scale_out

    margin_f = npz["ref_logits"][:, 1] - npz["ref_logits"][:, 0]
    margin_q = dequant[:, 1] - dequant[:, 0]
    pred = margin_q > 0
    ref = margin_f > 0
    agree_mask = pred == ref
    agreement = agree_mask.mean()
    err = np.abs(dequant - npz["ref_logits_int8sim"]).max()

    if not agree_mask.all():
        flipped = np.where(~agree_mask)[0]
        print("  int8 decision flips on %d/%d samples:" % (len(flipped), len(pred)))
        for i in flipped:
            print("    idx %3d: float margin %+.3f -> int8 margin %+.3f" %
                  (i, margin_f[i], margin_q[i]))

    return agreement >= min_agreement, agreement, err


def main():
    npz, meta, arch_dims, n_layers = load_sources()

    write_model_f32(npz, arch_dims, n_layers, OUT_F32_BIN)
    act_scales = write_model_int8(npz, meta, arch_dims, n_layers, OUT_INT8_BIN)
    write_shared_meta(meta, act_scales, OUT_META)

    ok_f32, err_f32 = self_check_f32(npz, OUT_F32_BIN)
    ok_int8, agree_int8, err_int8 = self_check_int8(npz, OUT_INT8_BIN, act_scales[-1])

    n_params = sum(npz[f"W{i}"].size + npz[f"b{i}"].size for i in range(n_layers))
    print("architecture:", meta["architecture"])
    print("params:", n_params)
    print("model_f32.bin: %.1f KB" % (os.path.getsize(OUT_F32_BIN) / 1024))
    print("model_int8.bin: %.1f KB" % (os.path.getsize(OUT_INT8_BIN) / 1024))
    print("threshold (logit diff):", meta["logit_diff_threshold"])
    print("f32 self-check: %s (max abs err %.2e)" % ("PASS" if ok_f32 else "FAIL", err_f32))
    print("int8 self-check: %s (decision agreement %.3f, dequant max err %.3f)" %
          ("PASS" if ok_int8 else "FAIL", agree_int8, err_int8))


if __name__ == "__main__":
    main()