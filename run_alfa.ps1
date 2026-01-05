$MODEL_NAME = "MTCL"
$DATA_TYPE = "FD"
$NUM_NODES = 13
$SEQ_LEN = 96
$PRED_LEN = 96
$BATCH_SIZE = 128
$TRAIN_EPOCHS = 15
$LEARNING_RATE = 0.0001
$DATA_ROOT = "./dataset/FD/"
if (-not (Test-Path $DATA_ROOT)) {
    Write-Host "Warning: Data directory $DATA_ROOT does not exist." -ForegroundColor Yellow
}
Write-Host "Starting Training with Feature Dimension: $NUM_NODES" -ForegroundColor Green
python -u run.py `
    --is_training 1 `
    --model $MODEL_NAME `
    --data $DATA_TYPE `
    --root_path $DATA_ROOT `
    --seq_len $SEQ_LEN `
    --pred_len $PRED_LEN `
    --d_model 16 `
    --d_ff 16 `
    --num_nodes $NUM_NODES `
    --layer_nums 4 `
    --k 2 `
    --num_experts_list 4 4 4 4 `
    --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 6 4 2 8 `
    --anomaly_ratio 3 `
    --batch_size $BATCH_SIZE `
    --train_epochs $TRAIN_EPOCHS `
    --learning_rate $LEARNING_RATE `
    --patience 10 `
    --revin 1 `
    --use_gpu True `
    --gpu 0