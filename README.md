#SEA-LION GPTQ quantization method
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

```python

```

# Benchmark

| Model                                        | ARC   | HellaSwag | MMLU  | TruthfulQA | Average |
| -------------------------------------------- | ----- | --------- | ----- | ---------- | ------- |
| SEA-LION 7B Instruct (FP16)                  | 40.78 | 68.20     | 27.12 | 36.29      | 43.10   |
| SEA-LION 7B Instruct (4-Bit, 128 group size) |  |      |  |       |    |

Although the evaluations were run with the same n-shot values as Hugging Face's LLM Leaderboard, the evaluations were run using version 0.4.1 of the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.1) by EleutherAI. AWQ support in the [VLLM](https://github.com/vllm-project/vllm/tree/v0.2.6) inference engine was used to perform the evaluations in the harness. If you wish to run the evaluations yourself, please setup VLLM using the following [instructions](https://github.com/aisingapore/sealion/tree/vllm/vllm).

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
