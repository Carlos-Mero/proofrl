python main.py \
  eval \
  --proof_model deepseek-v3.1-250821 \
  --eval_model deepseek-v3.1-250821 \
  --eval_dataset nproof/valid.json \
  --method proofrl \
  --prover_base_url <add_your_url_here> \
  --eval_base_url <add_your_url_here> \
  --api_key <add_your_api_key_here>
