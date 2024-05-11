# SEA-LION AWQ quantization method
## 1.Purpose

This repository provides a guide and a collection of scripts to help with the quantization and inference of the [SEA-LION 7B Instruct Model](https://huggingface.co/aisingapore/sea-lion-7b-instruct) instruct model developed by AI Singapore. The goal is to further democratise access to SEA-LION by allowing it to run on consumer grade hardware (e.g. common GPU like Nvidia GTX and RTX series) thanks to quantization.

The 4-bit, 128 group size quantized model can be found [here]().

## 2. Quantization 
Both quantization and inference is handled by the [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) library. Some custom patches are required for it work however, so a separate forked is being maintained over [here](https://github.com/caviato/AutoAWQ). 

Additionally, the file ffn.py from the SEA-LION instruct model needs to be replaced with the version provided in this repository. The provided AWQ model will already contain the correct ffn.py file. 

In the `quantize.py` file, please change the value of the two following variables to the appropriate path for your system.

```python
# quantize.py

# ...
base_model_path = "path/to/base"
quant_mode_path = "path/to/quant"
#...
```
Setting up the quantization config, loading the model via AutoAWQ and loading the tokenizer.
```python
# quantize.py

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

#...
# Quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
    }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
        base_model_path,
        **{"low_cpu_mem_usage": True}
        )
tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
        )
```
The AWQ algorithm requires some input data. Due to the multilingual nature of SEA-LION, we used data from each language available in SEA-LION.
```python
# quantize.py

from datasets import load_dataset

# ...
# Load dataset
lang_list = [ "en", "km", "ms", "ta", "tl", "zh", "id", "lo", "my", "th", "vi" ]
paths = []

for lang in lang_list:
    path = "/wikitexts_sealion/" + lang + "_wiki_cirrus.jsonl"
    paths.append(path)
    print(f"Added path {path}")

dataset = []
n_samples = 16
print("Generating samples from data...")
for path in paths:
    print(f"Loading data from {path}")
    data = load_dataset("json", data_files=path, split="train")
    print("Data loaded")

    text = data["text"]
    entries = 0
    fails = 0
    while(True):
        if entries == n_samples:
            print("Added to samples")
            break

        if fails == 100:
            print("Failed too many times")
            print(f"entries: {entries}")
            break

        if text[entries].strip == "":
            fails += 1
            continue

        dataset.append(text[entries])

        entries += 1
print("Calibration dataset prepared")
#...
```
Finally, we can quantize and save our model.
```python
# quantize.py

#...
# Quantize
model.to("cuda:0")
model.quantize(tokenizer, quant_config=quant_config, calib_data=dataset)

# Save quantized model
model.save_quantized(quant_mode_path, safetensors=True)
tokenizer.save_pretrained(quant_mode_path)
```
# Inference
We will be using [VLLM](https://github.com/vllm-project/vllm/tree/v0.2.6) to run inference. Please setup VLLM in your Python environment using the following [instructions](https://github.com/aisingapore/sealion/tree/vllm/vllm).

Set the path to the quantized model directory. 
```python
quant_mode_path = "path/to/quant"
```
Import vllm and initialise the model. 
```python
import math
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

#...
# Create an LLM.
llm = LLM(
        model=quant_mode_path,
        trust_remote_code=True,
        quantization="AWQ",
        )
#...
```
Create a sampling params object.
```python
#...
d_model = 4096

def scale_logits(token_ids, logits):
    logits = logits * (1 / math.sqrt(d_model))
    return logits

# 
sampling_params = SamplingParams(
    temperature=0,
    repetition_penalty=1.2,
    logits_processors=[scale_logits],
    max_tokens=64,
)
#...
```
Finally, run the inference with the prompt.
```python
# ...
# Sample prompts.
prompts = [
    "Hello, my name is John and I am a",
    "Singapore is",
]

# Generate text
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
# Benchmark

| Model                                        | ARC   | HellaSwag | MMLU  | Average |
| -------------------------------------------- | ----- | --------- | ----- | ------- |
| SEA-LION 7B Instruct (FP16)                  | 40.78 | 68.20     | 27.12 | 45.37   |
| SEA-LION 7B Instruct (4-Bit, 128 group size) | 37.97 | 63.68     | 28.00 | 43.22   |

TruthfulQA is excluded from the benchmarks due to a technical issue with the quantized model and the evaluation harness. 
Although the evaluations were run with the same n-shot values as Hugging Face's LLM Leaderboard, the evaluations were run using version 0.4.1 of the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.1) by EleutherAI. AWQ support in the [VLLM](https://github.com/vllm-project/vllm/tree/v0.2.6) inference engine was used to perform the evaluations in the harness. If you wish to run the evaluations yourself, please setup VLLM using the instructions found in the [inference section](#inference).

| Tasks                       | n-shots |
| --------------------------- | ------- |
| ARC (arc_challenge)         | 25      |
| HellaSwag (hellaswag)       | 10      |
| MMLU (mmlu)                 | 5       |
| TruthfulQA (truthfulqa_mc2) | 0       |

# Work In Progress (WIP)

- [ ] Inference time comparisons on A100
- [ ] Inference time of quantized model on GTX1070 (8GB)
- [ ] Inference time of quantized model on RTX3080 (10GB)

# Acknowledgements

Thank you to the AI Singapore team for their guidance and resources, with special thanks to:

- Ng Boon Cheong Raymond
- Teng Walter
- Siow Bryan
