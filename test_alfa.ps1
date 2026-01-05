
$STD_DATA_ROOT = "./dataset/FD/"

python -u run.py `
    --is_training 0 `
    --model MTCL `
    --data FD `
    --root_path $STD_DATA_ROOT `
    --seq_len 96 `
    --pred_len 96 `
    --d_model 16 `
    --d_ff 16 `
    --num_nodes 13 `
    --layer_nums 4 `
    --k 2 `
    --num_experts_list 4 4 4 4 `
    --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 6 4 2 8 `
    --anomaly_ratio 3 `
    --batch_size 128 `
    --train_epochs 15 `
    --learning_rate 0.001 `
    --patience 10 `
    --revin 1 `
    --use_gpu True `
    --gpu 0