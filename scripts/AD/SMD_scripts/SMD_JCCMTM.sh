export CUDA_VISIBLE_DEVICES=0

model_name='JCCMTM'
dset='SMD'

sparse_id='Null'
iters=1
pretrain_bsz=64
finetune_bsz=128
test_bsz=128

root_path='/home/JCCMTM/'
data_path='Datas/SMD/'

## JCCMTM config ##
e_layers=2
d_layers=0
kernel_size=3
mul_uni_ratio=1
lambda=0.1
strategy='CICD'
anomaly_ratio=0.5

d_ff=128
d_model=64
n_heads=4

## optim config ##
learning_rate_pre=1e-4
learning_rate_ft=1e-4
lradj='type1'

# # make pretrain data
# python  -u ./data_provider/make_data_pretrain.py \
#     --cfg 'SMD_pretrain-100.yaml' \
#     --dset $dset \
#     --stage 'pretrain' \
#     --task 'AD' \
#     --decomposition \
#     --kernel_size $kernel_size \
#     --category 'train' \
#     --root_path $root_path \
#     --data_path $data_path \

# # pretrain
# python -u run_pretrain.py \
#     --cfg 'SMD_pretrain-100.yaml' \
#     --dset $dset \
#     --task 'AD' \
#     --learning_rate $learning_rate_pre \
#     --train_epochs 3 \
#     --mem_len 0 \
#     --mem_len_uni 0 \
#     --e_layers $e_layers \
#     --d_layers $d_layers \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --n_heads $n_heads \
#     --load_sparse \
#     --sparse_id 'Null' \
#     --w_h $lambda \
#     --batch_size $pretrain_bsz \
#     --decomposition \
#     --kernel_size $kernel_size \
#     --root_path $root_path \
#     --model $model_name \

# end-to-end finetune
# pretrained_modelID='Null'

# python -u run_anomaly_detection.py \
#     --cfg 'SMD_finetune-100.yaml' \
#     --dset $dset \
#     --data $dset \
#     --root_path $root_path \
#     --data_path $data_path \
#     --task 'AD' \
#     --category 'train' \
#     --e_layers $e_layers \
#     --d_layers $d_layers \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --n_heads $n_heads \
#     --mem_len 0 \
#     --mem_len_uni 0 \
#     --load_sparse \
#     --sparse_id 'Null' \
#     --pretrain_model_id $pretrained_modelID \
#     --kernel_size $kernel_size \
#     --model $model_name \
#     --anomaly_ratio $anomaly_ratio \
#     --learning_rate 3e-4 \
#     --batch_size $finetune_bsz \
#     --test_batch_size $test_bsz \
#     --train_epochs 3 \
#     --decomposition \
