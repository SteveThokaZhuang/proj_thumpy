from safetensors import safe_open
import numpy as np
A="/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
for run in ("g1_mixnorm_sft","g1_mixnorm_s7_sft"):
    with safe_open(f"{A}/{run}/saves/adapter_model.safetensors", framework="np") as f:
        ks=list(f.keys()); na=nb=0; sa=sb=0.0
        print(f"=== {run}: {len(ks)} tensors ===")
        for k in ks:
            t=f.get_tensor(k).astype(np.float64); s=float((t**2).sum())
            if ".lora_A." in k: na+=1; sa+=s
            elif ".lora_B." in k: nb+=1; sb+=s
        print(f"  lora_A: {na} 个   ‖A‖={np.sqrt(sa):.4f}")
        print(f"  lora_B: {nb} 个   ‖B‖={np.sqrt(sb):.4f}")
        for k in ks[:4]:
            t=f.get_tensor(k)
            print(f"    {k:<56} {str(t.shape):<13} ‖·‖={np.linalg.norm(t):.4f} nz={np.count_nonzero(t)}/{t.size}")
