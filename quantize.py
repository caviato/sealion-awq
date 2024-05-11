from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

base_model_path = "path/to/base"
quant_mode_path = "path/to/quant"

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

# Quantize
model.to("cuda:0")
model.quantize(tokenizer, quant_config=quant_config, calib_data=dataset)

# Save quantized model
model.save_quantized(quant_mode_path, safetensors=True)
tokenizer.save_pretrained(quant_mode_path)

