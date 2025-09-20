accelerate launch \
  main.py \
  train \
  --method proofrl \
  --eval_model gemini-2.5-flash \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset nproof/train.json \
  --learning_rate 1e-6 \
  --dtype bfloat16 \
  --max_prompt_length 2048 \
  --max_completion_length 8192 \
  --log_completions \
  --per_device_train_batch_size 1 \
  --num_generations 4 \
  --importance_sampling_level sequence \
  --epsilon 3e-4 \
  --epsilon_high 4e-4 \
  --beta 0.0 \
  --loss_type grpo \
  --gradient_accumulation_steps 2 \
  --eval_base_url <add_your_url_here> \
  --api_key <add_your_api_key_here> \
  --num_train_epochs 1.0
