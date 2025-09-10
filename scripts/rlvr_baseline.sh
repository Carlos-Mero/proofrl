accelerate launch \
  --config_file dsconfig/dss3_config.yaml \
  main.py \
  train \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset HuggingFaceH4/MATH-500 \
  --learning_rate 1e-5 \
  --dtype bfloat16 \
  --max_prompt_length 2048 \
  --max_completion_length 2048 \
  --log_completions \
  --per_device_train_batch_size 2 \
  --num_generations 4 \
  --importance_sampling_level sequence \
  --epsilon 3e-4 \
  --epsilon_high 4e-4 \
  --beta 0.0 \
  --loss_type grpo \
  --gradient_accumulation_steps 1 \
