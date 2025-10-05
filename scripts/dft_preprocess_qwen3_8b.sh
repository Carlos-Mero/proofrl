accelerate launch main.py \
  train \
  --preprocess \
  --dataset /mnt/workspace/hyx/proofrl/eval_logs/1005T0740/samples.json \
  --model_name_or_path Qwen/Qwen3-8B-Base \
  --dtype bfloat16 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 1.0
