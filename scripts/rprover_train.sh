python main.py \
  eval \
  --proof_model qwen3-8b \
  --eval_model qwen3-30b-a3b \
  --eval_dataset NP_dataset/train_300.json \
  --method rprover \
  --ragtrain \
  --prover_base_url <add_your_url_here> \
  --eval_base_url <add_your_url_here> \
  --api_key <add_your_api_key_here> \
  --rag_url <add_your_lightrag_url_here>
