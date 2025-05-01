
#!/bin/bash
echo "Starting fine-tuning with nohup..."
nohup python mortgage_finetune/train_lora_v2.py > training.log 2>&1 &
echo "Training started. Monitor with: tail -f training.log"
