import importlib
import sys

import torch


def main() -> int:
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print(f"gpu_name={torch.cuda.get_device_name(device_index)}")
        print(f"gpu_capability={torch.cuda.get_device_capability(device_index)}")

    for module_name in ["flash_attn", "apex", "apex.normalization"]:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"import {module_name}=ok version={version}")
        except Exception as exc:
            print(f"import {module_name}=failed {type(exc).__name__}: {exc}")

    if not torch.cuda.is_available():
        return 1

    try:
        from flash_attn import flash_attn_varlen_func

        q = torch.randn(8, 2, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(8, 2, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(8, 2, 64, device="cuda", dtype=torch.float16)
        cu_seqlens = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 8, 8)
        torch.cuda.synchronize()
        print(f"flash_attn_varlen_func=ok shape={tuple(output.shape)}")
    except Exception as exc:
        print(f"flash_attn_varlen_func=failed {type(exc).__name__}: {exc}")
        return 2

    try:
        from apex.normalization import FusedRMSNorm

        norm = FusedRMSNorm(normalized_shape=64).cuda().half()
        value = torch.randn(4, 64, device="cuda", dtype=torch.float16)
        output = norm(value)
        torch.cuda.synchronize()
        print(f"apex_FusedRMSNorm=ok shape={tuple(output.shape)}")
    except Exception as exc:
        print(f"apex_FusedRMSNorm=failed {type(exc).__name__}: {exc}")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())