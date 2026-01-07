$MODEL_NAME = "MTCL"
$DATA_TYPE = "ALFA_ad"  # set to "ALFA_ad" for ALFA
$NUM_NODES = 13
$SEQ_LEN = 96
$PRED_LEN = 96
$BATCH_SIZE = 128
$TRAIN_EPOCHS = 20
$LEARNING_RATE = 0.0003
$DATA_ROOT = if ($DATA_TYPE -eq "ALFA_ad") { "./dataset/ALFA/" } else { "./dataset/FD/" }

# Diffusion toggles (set $USE_DIFFUSION to $true to enable)
$USE_DIFFUSION = $true
$LAMBDA_DIFF = 1.0
$LAMBDA_REC = 0.05
$DIFF_STEPS = 60
$DIFF_EVAL_STEP = 30
$DIFF_DIM = 96
$DIFF_HEADS = 6
$DIFF_DEPTH = 3
$DIFF_SCHEDULE = "cosine"
$DIFF_BETA_START = 0.0001
$DIFF_BETA_END = 0.02
$DIFFUSION_SAMPLING = "ddim"
$DDIM_STEPS = 20
$DDIM_ETA = 0.0
$ACCUM_STEPS = 2
$THRESHOLD_MODE = "train_test" # options: train_test | test_only | val_test
$USE_NOISE_SCORE = $true        # use single-step noise score
$USE_EVT = $false               # disable EVT thresholding (use percentile)
$EVT_TAIL_FRAC = 0.02
$EVT_CONF = 0.95
$ANOMALY_RATIO = 4
$NOISE_SCORE_MODE = "ebm"      # l2 | ebm | knn
$USE_EBM = $true
$LAMBDA_EBM = 0.1
$EBM_HIDDEN = 64
$KNN_K = 5
$KNN_MAX_SAMPLES = 50000
$USE_GRAPH = $true
$GRAPH_DYNAMIC = $true
$GRAPH_EMB_DIM = 8
$GRAPH_DROPOUT = 0.0
$GRAPH_ALPHA = 0.2

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
        "--diffusion_beta_end", $DIFF_BETA_END,
        "--diffusion_sampling", $DIFFUSION_SAMPLING,
        "--ddim_steps", $DDIM_STEPS,
        "--ddim_eta", $DDIM_ETA,
        "--accum_steps", $ACCUM_STEPS,
        "--noise_score_mode", $NOISE_SCORE_MODE
    )
    if ($USE_NOISE_SCORE) {
        $diffArgs += @("--use_noise_score")
    }
    if ($USE_EVT) {
        $diffArgs += @("--use_evt", "--evt_tail_frac", $EVT_TAIL_FRAC, "--evt_conf", $EVT_CONF)
    }
    if ($USE_EBM) {
        $diffArgs += @("--use_ebm", "--lambda_ebm", $LAMBDA_EBM, "--ebm_hidden", $EBM_HIDDEN)
    }
    if ($NOISE_SCORE_MODE -eq "knn") {
        $diffArgs += @("--knn_k", $KNN_K, "--knn_max_samples", $KNN_MAX_SAMPLES)
    }
    if ($USE_GRAPH) {
        $diffArgs += @("--use_graph", "--graph_emb_dim", $GRAPH_EMB_DIM, "--graph_dropout", $GRAPH_DROPOUT, "--graph_alpha", $GRAPH_ALPHA)
        if ($GRAPH_DYNAMIC) {
            $diffArgs += @("--graph_dynamic")
        }
    }
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
    --anomaly_ratio $ANOMALY_RATIO `
    --batch_size $BATCH_SIZE `
    --train_epochs $TRAIN_EPOCHS `
    --learning_rate $LEARNING_RATE `
    --patience 5 `
    --revin 1 `
    --use_gpu True `
    --gpu 0 `
    --threshold_mode $THRESHOLD_MODE `
    @diffArgs
