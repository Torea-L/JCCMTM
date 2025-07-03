export CUDA_VISIBLE_DEVICES=0

model_name='JCCMTM'

sparse_id='Null'
iters=1
pretrain_bsz=32
finetune_bsz=128
test_bsz=128


root_path='/home/JCCMTM/'
data_path='Datas/MSL/'


## JCCMTM config ##
e_layers=2
d_layers=0
d_model=16
d_ff=16
n_heads=4
kernel_size=15
mul_uni_ratio=1
lambda=1.0
strategy='CICD'
anomaly_ratio=1.0

## optim config ##
learning_rate_pre=1e-3
learning_rate_ft=1e-4
lradj='type1'

# # make pretrain data
# python  -u ./data_provider/make_data_pretrain.py \
#     --cfg 'MSL_pretrain-100.yaml' \
#     --dset MSL \
#     --stage 'pretrain' \
#     --task 'AD' \
#     --decomposition \
#     --kernel_size $kernel_size \
#     --category 'train' \
#     --root_path $root_path \
#     --data_path $data_path \

# # pretrain
# python -u run_pretrain.py \
#     --cfg 'MSL_pretrain-100.yaml' \
#     --dset MSL \
#     --task 'AD' \
#     --learning_rate $learning_rate_pre \
#     --train_epochs 3 \
#     --mem_len 0 \
#     --mem_len_uni 0 \
#     --e_layers $e_layers \
#     --d_layers $d_layers \
#     --d_model $d_model\
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

# # end-to-end finetune
# pretrained_modelID = 'Null'
# python -u run_anomaly_detection.py \
#     --cfg 'MSL_finetune-100.yaml' \
#     --dset MSL \
#     --data MSL \
#     --root_path $root_path \
#     --data_path $data_path \
#     --task 'AD' \
#     --category 'train' \
#     --mem_len 0 \
#     --mem_len_uni 0 \
#     --e_layers $e_layers \
#     --d_layers $d_layers \
#     --d_model $d_model\
#     --d_ff $d_ff \
#     --n_heads $n_heads \
#     --load_sparse \
#     --sparse_id 'Null' \
#     --decomposition \
#     --pretrain_model_id $pretrained_modelID \
#     --kernel_size $kernel_size \
#     --model $model_name \
#     --anomaly_ratio $anomaly_ratio \
#     --learning_rate $learning_rate_ft \
#     --batch_size $finetune_bsz \
#     --test_batch_size $test_bsz \
#     --train_epochs 5 \
