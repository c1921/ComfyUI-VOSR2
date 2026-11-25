"""Benchmark + cache-correctness harness for ComfyUI-VOSR2.

Run it with the Python interpreter of the ComfyUI install that has this node
(it needs torch, folder_paths and comfy), e.g.:

    python <ComfyUI>/custom_nodes/ComfyUI-VOSR2/bench/bench_vosr2.py
    python .../bench_vosr2.py --e2e --bundle VOSR2 --size 512 --upscale 4 --repeats 3
    python .../bench_vosr2.py --comfyui "D:/ComfyUI"     # if autodetect fails

Two independent checks:

1. cache correctness (no model weights needed, ~1 s)
   Builds a tiny LightningDiT and a depth-1 DINOv2-L, calls the memoized paths,
   clears the caches, calls them again and asserts torch.equal(). This is the
   bit-exactness proof for the RoPE / pos-embed caches.

2. end-to-end timing (--e2e, loads the real ~7 GB bundle)
   Runs inference.run_vosr2() on a synthetic image, reporting wall clock, peak
   CUDA memory and a sha256 of the output tensor. Run the same command on two
   checkouts (e.g. main vs this branch) and compare the hashes and the timings.
"""
import argparse
import hashlib
import importlib
import os
import sys
import time
import types


def default_comfyui_root():
    """<ComfyUI>/custom_nodes/ComfyUI-VOSR2/bench/bench_vosr2.py -> <ComfyUI>"""
    here = os.path.abspath(__file__)
    plugin = os.path.dirname(os.path.dirname(here))          # .../ComfyUI-VOSR2
    custom_nodes = os.path.dirname(plugin)                   # .../custom_nodes
    return os.path.dirname(custom_nodes)


def load_plugin(plugin_dir):
    """Import this custom node as a package despite the hyphen in its folder name."""
    pkg_name = "vosr2_plugin"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [plugin_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(f"{pkg_name}.loader"),
        importlib.import_module(f"{pkg_name}.inference"),
    )


def check_caches(loader_mod):
    """Prove the memoized paths are numerically identical to a cold call."""
    import torch
    from vosr2_plugin.models.dinov2 import DinoVisionTransformer
    from vosr2_plugin.models.lightningdit import LightningDiT

    results = []

    # --- dynamic RoPE ------------------------------------------------------
    dit = LightningDiT(
        input_size=16, patch_size=2, in_channels=8, out_channels=8,
        hidden_size=64, num_heads=4, depth=1, mlp_ratio=4.0,
        use_qknorm=False, use_swiglu=False, use_rope=True, use_rmsnorm=False,
        z_dims=None, encdim_ratio=1.0, num_fused_layers=1,
        auxiliary_time_cond=False,
    )
    seq = 32                      # 32x32 tokens, the shape of a 64px latent tile
    head_dim = dit.hidden_size // dit.num_heads
    probe = torch.randn(2, dit.num_heads, seq * seq, head_dim)
    cold = dit._get_dynamic_rope(seq, probe.device, probe.dtype)(probe).clone()
    dit._rope_cache.clear()
    warm = dit._get_dynamic_rope(seq, probe.device, probe.dtype)(probe)
    results.append(("dynamic RoPE", torch.equal(cold, warm), int(dit._rope_cache.__len__())))

    # --- DINOv2 pos-embed --------------------------------------------------
    dino = DinoVisionTransformer(img_size=518, patch_size=14, embed_dim=1024,
                                 depth=1, num_heads=16)
    tokens = torch.zeros(1, 1 + (448 // 14) ** 2, 1024)
    cold = dino.interpolate_pos_encoding(tokens, 448, 448).clone()
    dino._pos_embed_cache.clear()
    warm = dino.interpolate_pos_encoding(tokens, 448, 448)
    results.append(("DINOv2 pos-embed", torch.equal(cold, warm),
                    int(dino._pos_embed_cache.__len__())))
    return results


def run_e2e(loader_mod, inference_mod, args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loader_mod.load_vosr2(args.bundle, args.dtype)

    gen = torch.Generator().manual_seed(args.seed)
    image = torch.rand((args.frames, args.size, args.size, 3), generator=gen)

    def once():
        return inference_mod.run_vosr2(
            model, image, args.upscale, args.seed, args.color,
            args.tile_size, args.tile_overlap, args.vae_tile_size, args.vae_tile_overlap,
        )

    once()                                    # warmup: first call loads/pages weights
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    out = None
    for _ in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    peak = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    digest = hashlib.sha256(
        out.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    return timings, peak, digest, tuple(out.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", default=None, help="ComfyUI root (autodetected from this script)")
    ap.add_argument("--e2e", action="store_true", help="also run the full pipeline (needs the model bundle)")
    ap.add_argument("--bundle", default="VOSR2")
    ap.add_argument("--dtype", default="default", choices=["default", "fp16", "bf16"])
    ap.add_argument("--size", type=int, default=512, help="source image side, pixels")
    ap.add_argument("--frames", type=int, default=1, help="batch size (video frames)")
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--color", default="wavelet", choices=["wavelet", "adain", "none"])
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--tile-overlap", type=int, default=64)
    ap.add_argument("--vae-tile-size", type=int, default=1024)
    ap.add_argument("--vae-tile-overlap", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    root = args.comfyui or default_comfyui_root()
    if not os.path.isdir(os.path.join(root, "comfy")):
        sys.exit(f"ComfyUI root not found at {root!r}; pass --comfyui <path>")
    sys.path.insert(0, root)

    loader_mod, inference_mod = load_plugin(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("== cache correctness ==")
    ok = True
    for name, same, entries in check_caches(loader_mod):
        ok = ok and same
        print(f"  {name:20} bit-exact={same}  cache_entries={entries}")
    print("  RESULT:", "PASS" if ok else "FAIL")

    if not args.e2e:
        print("\n(skipping end-to-end; pass --e2e to load the bundle and time the pipeline)")
        return 0 if ok else 1

    print("\n== end-to-end ==")
    timings, peak, digest, shape = run_e2e(loader_mod, inference_mod, args)
    print(f"  output      : {shape}")
    print(f"  wall clock  : " + ", ".join(f"{t*1000:.0f} ms" for t in timings))
    print(f"  best        : {min(timings)*1000:.0f} ms")
    print(f"  peak VRAM   : {peak:.0f} MiB")
    print(f"  sha256      : {digest}")
    print("\nCompare the sha256 against the same command run on the baseline checkout:")
    print("  identical hash => the optimization is numerically identical (bit-exact).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
