model_loc=
model_base_name=sea-lion-7b-instruct
model_quant_name=

output_path=
result_quant_name=

model_quant_path="${model_loc}${model_quant_name}"

task_name="arc_challenge"
n_shot=25

lm_eval --model vllm \
        --model_args pretrained=${model_quant_path},quantization=awq,trust_remote_code=True \
        --tasks ${task_name} \
        --num_fewshot ${n_shot} \
        --output_path "${output_path}${result_quant_name}"

