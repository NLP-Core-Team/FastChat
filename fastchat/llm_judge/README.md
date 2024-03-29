# ruMT-Bench: Benchmarking of Multi-Turn Russian Alignment of Large Language Models

You can use ruMT-bench in Russian language prompts to evaluate your models with LLM-as-a-judge.

## Dataset

- [ruMT-Bench Dataset](https://huggingface.co/datasets/NLPCoreTeam/ruMT-Bench)

The dataset contains instructive multi-turn open-ended questions for evaluating chat assistants, divided into 8 topics: writing, roleplay, extraction, reasoning, math, coding, knowledge I (STEM), and knowledge II (humanities/social science). The English version of MT-bench has been translated into Russian.

## Getting started

To install release, run

```
git clone https://github.com/NLP-Core-Team/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

## Review Pre-Generated Model Answers and Judgments
We provide pre-generated model answers and judgments for some models.

To download the pre-generated data, use
```
python download_mt_bench_pregenerated.py
```

## ruMT-Bench

### Evaluate a model on ruMT-bench

#### Step 1. Generate model answers to ruMT-bench questions
```
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID] --bench-name [BENCH-NAME]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.
  - `[BENCH-NAME]` is benchmark name ru_mt_bench or mt_bench.

e.g.,
```
python gen_model_answer.py --model-path mistralai/Mixtral-8x7B-v0.1 --model-id Mixtral-8x7B-v0.1 --bench-name ru_mt_bench
python gen_model_answer.py --model-path IlyaGusev/saiga_mistral_7b --model-id saiga_mistral_7b --bench-name ru_mt_bench
```
The answers will be saved to `data/ru_mt_bench/model_answer/[MODEL-ID].jsonl`.

To make sure FastChat loads the correct prompt template, see the supported models and how to add a new model [here](../../docs/model_support.md#how-to-support-a-new-model).

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In ruMT-bench we recommend and have ourselves evaluated single-answer grading using reference responses.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --bench-name [BENCH-NAME]
```

e.g.,
```
python gen_judgment.py --model-list Mixtral-8x7B-v0.1 gemini-pro gpt-3.5-turbo gpt-4 --parallel 20 --bench-name ru_mt_bench
```
The judgments will be saved to `data/ru_mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```
  python show_result.py --model-list Mixtral-8x7B-v0.1 gemini-pro gpt-3.5-turbo gpt-4 --bench-name ru_mt_bench
  ```
- Show all scores
  ```
  python show_result.py --bench-name ru_mt_bench
  ```

---

### Other grading options
Besides score-based single-answer grading, we also support two additional grading options based on win rates:
- `pariwise-baseline`: run pairwise comparison against a baseline model.
- `pairwise-all`: run pairwise comparison between all model pairs on all questions.

#### Option 2: pairwise comparison against a baseline (default: gpt-3.5-turbo)

- Generate GPT-4 judgments
```
python gen_judgment.py --mode pairwise-baseline --model-list Mixtral-8x7B-v0.1 gemini-pro gpt-3.5-turbo gpt-4 --parallel 20 --bench-name ru_mt_bench
```
The judgments will be saved to `data/ru_mt_bench/model_judgment/gpt-4_pair.jsonl`

- Show results
```
python show_result.py --mode pairwise-baseline --bench-name ru_mt_bench
```

#### Option 3: Run GPT-4 judge with all pair comparisons

Another option is to run pairwise comparisons on all possible pairs.
This could be more expensive when #models increases, but it gives you a more comprehensive information.

```
python gen_judgment.py --mode pairwise-all --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --bench-name [BENCH-NAME]
```

```
python show_result.py --mode pairwise-all --bench-name ru_mt_bench
```

### How to get GPT-4/GPT-3.5/Gemini/Gigachat's answer?

Generate GPT-3.5/4 answers.
```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_api_answer.py --model gpt-4 --bench-name ru_mt_bench
```
Generate Gemini answers.
```
export GEMINI_API_KEY=XXXXXX  # set the Gemini API key
python gen_api_answer.py --model gemini-pro --bench-name ru_mt_bench
```
Generate GigaChat answers.
```
export GIGACHAT_API_KEY=XXXXXX  # set the GigaChat API key
python gen_api_answer.py --model GigaChat-Pro --bench-name ru_mt_bench
python gen_api_answer.py --model GigaChat:latest --bench-name ru_mt_bench
```

### How to plot the radar figure?

You can use this [colab notebook](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO) to plot the radar figure for MT-bench.

## Contributions
All evaluations, code adaptation and benchmark translation were done by the NLP core RnD team [Telegram channel](https://t.me/nlpcoreteam)