$MODEL_NAME = "MTCL"
$DATA_TYPE = "ALFA_ad"
$NUM_NODES = 13
$SEQ_LEN = 96
$PRED_LEN = 96
$BATCH_SIZE = 128
$TRAIN_EPOCHS = 12
$LEARNING_RATE = 0.001
$DATA_ROOT = "./dataset/ALFA/"

# Diffusion toggles (set $USE_DIFFUSION to $true to enable)
$USE_DIFFUSION = $true
$LAMBDA_DIFF = 1.0
$LAMBDA_REC = 0.0
$DIFF_STEPS = 100
$DIFF_EVAL_STEP = 50
$DIFF_DIM = 64
$DIFF_HEADS = 4
$DIFF_DEPTH = 3
$DIFF_SCHEDULE = "linear"
$DIFF_BETA_START = 0.0001
$DIFF_BETA_END = 0.02
$THRESHOLD_MODE = "val_test" # options: train_test | test_only | val_test

if (-not (Test-Path $DATA_ROOT)) {
    Write-Host "Warning: Data directory $DATA_ROOT does not exist." -ForegroundColor Yellow
}

Write-Host "Starting Training with Feature Dimension: $NUM_NODES" -ForegroundColor Green

$diffArgs = @()
if ($USE_DIFFUSION) {
    $diffArgs = @(
        "--use_diffusion",
        "--lambda_diff", $LAMBDA_DIFF,
        "--lambda_rec", $LAMBDA_REC,
        "--diffusion_steps", $DIFF_STEPS,
        "--diffusion_eval_step", $DIFF_EVAL_STEP,
        "--diffusion_dim", $DIFF_DIM,
        "--diffusion_heads", $DIFF_HEADS,
        "--diffusion_depth", $DIFF_DEPTH,
        "--diffusion_beta_schedule", $DIFF_SCHEDULE,
        "--diffusion_beta_start", $DIFF_BETA_START,
        "--diffusion_beta_end", $DIFF_BETA_END
    )
}

python -u run.py `
    --is_training 0 `
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
    --gpu 0 `
    --threshold_mode $THRESHOLD_MODE `
    @diffArgs
