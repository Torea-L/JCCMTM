export CUDA_VISIBLE_DEVICES=0

model_name=JCCMTM
pretrained_modelID='Null'

iters=1
train_bsz=128
test_bsz=256

root_path='/home/JCCMTM/'
data_path='Datas/ETT-small/ETTm1.csv'
strategy='CICD'

sparse_id='Null'

## JCCMTM config ##
e_layers=2
d_layers=0
kernel_size=15
mul_uni_ratio=1
lambda=1.0

## optim config ##
learning_rate_pre=1e-4
learning_rate_ft=5e-4
lradj='type1'

# make pretrain data
python -u ./data_provider/make_data_pretrain.py \
    --cfg 'ETTm_pretrain.yaml' \
    --dset 'ETTm1' \
    --stage 'pretrain' \
    --task 'LTSF' \
    --kernel_size $kernel_size \
    --category 'train' \
    --root_path $root_path \
    --decomposition \

# Pretrain
python -u run_pretrain.py \
    --cfg 'ETTm_pretrain.yaml' \
    --dset 'ETTm1' \
    --stage 'pretrain' \
    --task 'LTSF' \
    --learning_rate $learning_rate_pre \
    --load_sparse \
    --sparse_id 'Null' \
    --train_epochs 50 \
    --w_h $lambda \
    --kernel_size $kernel_size \
    --e_layers $e_layers \
    --root_path $root_path \
    --model $model_name \
    --strategy $strategy \
    --decomposition \
    --use_tb \

# Finetune
pretrained_modelID='2024-11-12-07_39_44'

for (( i=1; i<=$iters; i++ )) do
for pred_len in '96' '192' '336' '720'; do
# for pred_len in '96' ; do
    python -u run_long_term_forecasting.py \
            --cfg 'ETTm_finetune.yaml' \
            --dset 'ETTm1' \
            --data 'ETTm1' \
            --root_path $root_path \
            --data_path $data_path \
            --seq_len 96 \
            --pred_len $pred_len \
            --train_epochs 10 \
            --model $model_name \
            --batch_size $train_bsz \
            --test_batch_size $test_bsz \
            --num_workers 2 \
            --delta 0.001 \
            --learning_rate $learning_rate_ft \
            --lradj $lradj \
            --scale_adjustLR 1 \
            --pretrain_model_id $pretrained_modelID \
            --mul_uni_ratio $mul_uni_ratio \
            --e_layers $e_layers \
            --category 'train' \
            --load_sparse \
            --sparse_id $sparse_id \
            --strategy $strategy \
            --kernel_size $kernel_size \
            --decomposition \
    
done
done
