import math
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

quant_mode_path = "path/to/quant"

# Create an LLM.
llm = LLM(
        model=quant_model_path,
        trust_remote_code=True,
        quantization="AWQ",
        )

# Create a sampling params object.
d_model = 4096

def scale_logits(token_ids, logits):
    logits = logits * (1 / math.sqrt(d_model))
    return logits

sampling_params = SamplingParams(
    temperature=0,
    repetition_penalty=1.2,
    logits_processors=[scale_logits],
    max_tokens=64,
)

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

