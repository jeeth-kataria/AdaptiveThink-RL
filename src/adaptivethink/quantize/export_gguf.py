"""
Stage 3 quantisation: merge the router LoRA adapter into the base reasoner,
convert to GGUF, and quantise to Q4_K_M for on-device (llama.cpp) inference.

Steps:
  1. PEFT merge_and_unload -> a standalone FP16 HF checkpoint.
  2. llama.cpp convert_hf_to_gguf.py -> f16 GGUF.
  3. llama-quantize -> Q4_K_M GGUF.

Requires a local llama.cpp checkout (path via --llama-cpp or $LLAMA_CPP_DIR);
if missing it is cloned + built automatically.
"""
import argparse
import os
import subprocess
from pathlib import Path

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def merge_adapter(base_model: str, adapter_path: str, out_dir: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[quantize] merging adapter {adapter_path} into {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    print(f"[quantize] merged model -> {out_dir}")
    return out_dir


def ensure_llama_cpp(path: str | None) -> Path:
    path = path or os.environ.get("LLAMA_CPP_DIR", "third_party/llama.cpp")
    p = Path(path)
    if not p.exists():
        print(f"[quantize] cloning llama.cpp -> {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", str(p)], check=True)
    quant_bin = p / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        print("[quantize] building llama.cpp (cmake) ...")
        subprocess.run(["cmake", "-B", str(p / "build"), "-S", str(p)], check=True)
        subprocess.run(["cmake", "--build", str(p / "build"),
                        "--config", "Release", "-j"], check=True)
    return p


def convert_and_quantize(merged_dir: str, llama_dir: Path, out_gguf: str,
                         quant_type: str = "Q4_K_M") -> str:
    f16_path = str(Path(out_gguf).with_suffix(".f16.gguf"))
    Path(out_gguf).parent.mkdir(parents=True, exist_ok=True)

    convert = llama_dir / "convert_hf_to_gguf.py"
    print(f"[quantize] convert -> {f16_path}")
    subprocess.run(["python", str(convert), merged_dir,
                    "--outfile", f16_path, "--outtype", "f16"], check=True)

    quant_bin = llama_dir / "build" / "bin" / "llama-quantize"
    print(f"[quantize] quantize {quant_type} -> {out_gguf}")
    subprocess.run([str(quant_bin), f16_path, out_gguf, quant_type], check=True)
    print(f"[quantize] done: {out_gguf}")
    return out_gguf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--adapter", default="outputs/router-seed0",
                   help="router LoRA dir (omit with --no-merge for the raw base)")
    p.add_argument("--no-merge", action="store_true",
                   help="skip merge, quantise the base model directly")
    p.add_argument("--merged-dir", default="outputs/router-merged")
    p.add_argument("--llama-cpp", default=None)
    p.add_argument("--out", default="outputs/gguf/router-1p5b-Q4_K_M.gguf")
    p.add_argument("--quant-type", default="Q4_K_M")
    args = p.parse_args()

    src = args.base_model if args.no_merge else merge_adapter(
        args.base_model, args.adapter, args.merged_dir)
    llama_dir = ensure_llama_cpp(args.llama_cpp)
    convert_and_quantize(src, llama_dir, args.out, args.quant_type)


if __name__ == "__main__":
    main()
