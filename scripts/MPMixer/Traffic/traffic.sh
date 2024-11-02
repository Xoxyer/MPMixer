if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=MPMixer

root_path_name=./data/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --tra_layers 2 \
      --n_heads 16 \
      --d_model 256 \
      --d_ff 2048 \
      --num_down 3 \
      --dropout 0.05 \
      --fc_dropout 0.05 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'train' \
      --train_epochs 50\
      --itr 1 --batch_size 32 --learning_rate 0.0001
done