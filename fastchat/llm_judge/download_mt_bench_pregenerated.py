"""
Download the pre-generated model answers and judgments for MT-bench.
"""
import os

from fastchat.utils import run_cmd

filenames = [
    "model_answer/Mixtral-8x7B-Instruct-v0.1.jsonl",
    "model_answer/gemini-pro.jsonl",
    "model_answer/gemma-2b-it.jsonl",
    "model_answer/gemma-7b-it.jsonl",
    "model_answer/gpt-3.5-turbo-0125.jsonl",
    "model_answer/pt-3.5-turbo.jsonl",
    "model_answer/gpt-4-turbo-preview.jsonl",
    "model_answer/gpt-4.jsonl",
    "model_answer/saiga_mistral_7b.jsonl",
    "model_judgment/gpt-4_single.jsonl",
]


if __name__ == "__main__":
    prefix = "https://huggingface.co/datasets/NLPCoreTeam/ruMT-Bench/resolve/main/"

    for name in filenames:
        name_save = "data/ru_mt_bench/" + name
        os.makedirs(os.path.dirname(name_save), exist_ok=True)
        ret = run_cmd(f"wget -q --show-progress -O {name_save} {prefix + name}")
        assert ret == 0
