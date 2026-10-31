"""
Generate sample clean and tampered/stego neural network checkpoints
for testing Neural Guard.
"""

import os
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import save_file

def create_sample_models(output_dir: str = "sample_models"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # ─────────────────────────────────────────────────────────────────
    # 1. Clean Model (simulating a standard deep learning model)
    # ─────────────────────────────────────────────────────────────────
    clean_tensors_np = {
        "encoder.layer1.weight": rng.randn(128, 128).astype(np.float32) * 0.05,
        "encoder.layer1.bias": (rng.randn(128).astype(np.float32) * 0.01),
        "encoder.layer2.weight": rng.randn(256, 128).astype(np.float32) * 0.05,
        "encoder.layer2.bias": (rng.randn(256).astype(np.float32) * 0.01),
        "decoder.fc.weight": rng.randn(64, 256).astype(np.float32) * 0.05,
        "decoder.fc.bias": (rng.randn(64).astype(np.float32) * 0.01),
    }

    clean_safetensors_path = out_path / "clean_model.safetensors"
    save_file(clean_tensors_np, str(clean_safetensors_path))
    print(f"[+] Saved clean model: {clean_safetensors_path}")

    # PyTorch format
    clean_torch_path = out_path / "clean_model.pt"
    clean_tensors_torch = {k: torch.from_numpy(v) for k, v in clean_tensors_np.items()}
    torch.save(clean_tensors_torch, clean_torch_path)
    print(f"[+] Saved clean PyTorch model: {clean_torch_path}")

    # ─────────────────────────────────────────────────────────────────
    # 2. Tampered / Steganographic Model
    # ─────────────────────────────────────────────────────────────────
    tampered_tensors_np = {}
    for name, tensor in clean_tensors_np.items():
        if "weight" in name and tensor.size > 1000:
            # Embed steganographic high-entropy payload into mantissa LSBs
            flat = tensor.copy().flatten()
            bits = flat.view(np.uint32)
            n_modify = int(len(bits) * 0.6)  # embed in 60% of weights
            indices = rng.choice(len(bits), n_modify, replace=False)
            
            # Artificial random encrypted payload in 2 LSB bits
            num_bits = 2
            mask = (1 << num_bits) - 1
            payload_bits = rng.randint(0, mask + 1, n_modify, dtype=np.uint32)
            bits[indices] = (bits[indices] & ~np.uint32(mask)) | payload_bits
            
            tampered_tensor = bits.view(np.float32).reshape(tensor.shape)
            tampered_tensors_np[name] = tampered_tensor
        else:
            tampered_tensors_np[name] = tensor.copy()

    tampered_safetensors_path = out_path / "tampered_stego_model.safetensors"
    save_file(tampered_tensors_np, str(tampered_safetensors_path))
    print(f"[+] Saved tampered/stego model: {tampered_safetensors_path}")

    # PyTorch format
    tampered_torch_path = out_path / "tampered_stego_model.pt"
    tampered_tensors_torch = {k: torch.from_numpy(v) for k, v in tampered_tensors_np.items()}
    torch.save(tampered_tensors_torch, tampered_torch_path)
    print(f"[+] Saved tampered/stego PyTorch model: {tampered_torch_path}")

if __name__ == "__main__":
    create_sample_models()
