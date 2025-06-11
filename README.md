# JCCMTM
# Official implementation of **JCCMTM: Joint Channel-independent and Channel-dependent Strategy for Masked Multivariate Time-Series Modeling**


The experimental procedure consists of three key steps:

[1] ​Pre-training Data Preparation: The processed data is saved as .npy files for reusable storage upon completion;

​[2] Pre-training Phase;

​[3] Sub-task Fine-tuning.

======================================================

[1] Example Bash Command for Pre-training Data Preparation:​

'''
# make pretrain data
python -u ./data_provider/make_data_pretrain.py \
    --cfg 'ETTm_pretrain_crossdomain.yaml' \
    --dset 'ETTm2' \
    --stage 'pretrain' \
    --task 'LTSF' \
    --decomposition \
    --kernel_size $kernel_size \
    --category 'train' \
    --seq_len 336 \
    --root_path $root_path
'''

Once created, the pre-training data becomes reusable for multiple experiments. To generate new datasets, simply reconfigure the parameters and re-execute make_data_pretrain.py


=====================================================
[2] Pre-training:​

# Pretrain
python -u run_pretrain.py \
        --cfg 'ETTm_pretrain_crossdomain.yaml' \
        --dset 'ETTm2' \
        --stage 'pretrain' \
        --task 'LTSF' \
        --learning_rate $learning_rate_pre \
        --load_sparse \
        --sparse_id 'Null' \
        --train_epochs 50 \
        --seq_len 336 \
        --w_h $lambda \
        --kernel_size $kernel_size \
        --e_layers  $e_layers  \
        --root_path $root_path \
        --model $model_name \
        --strategy $strategy \
        --decomposition \


=====================================================
[3] Pre-training:​

# finetune
python -u run_long_term_forecasting.py \
        --cfg 'ETTm_finetune_crossdomain.yaml' \
        --dset 'ETTm1' \
        --data 'ETTm1' \
        --root_path $root_path \
        --data_path 'Datas/ETT-small/ETTm1.csv' \
        --seq_len 336 \
        --pred_len $pred_len \
        --train_epochs 10 \
        --model $model_name \
        --batch_size $train_bsz \
        --test_batch_size $test_bsz \
        --num_workers 2 \
        --delta 0.0 \
        --learning_rate $learning_rate_ft \
        --lradj $lradj \
        --scale_adjustLR 1 \
        --pretrain_model_id $pretrained_modelID \
        --pre_model_path $pretrained_modelID \
        --kernel_size $kernel_size \
        --mul_uni_ratio $mul_uni_ratio \
        --e_layers $e_layers \
        --category 'train' \
        --load_sparse \
        --sparse_id $sparse_id \
        --strategy $strategy \
        --decomposition \



===================================================

Key Parameter Specifications:​​


--​**strategy**:  Channel modeling strategy, options: ['CICD', 'CI', 'CD']


​--**decomposition**​ (bool):  Enables data pre-decomposition when True, generating trend and residual components

--component:  Input component for Uni-Module, options: ['trend', 'residual']; Only effective when decomposition=True

--kernel_size: Specifies the convolutional kernel size for average pooling when extracting 'trend' components. *It must be odd!!*


--​**load_sparse**​ (bool):  Enables sparse attention (default=True). Set --sparse_id='Null' to disable sparse masking.

--sparse_id:  Unique identifier for sparse attention matrix files


--​**efficient**​ (bool):  Activates JCCMTM-E when True

--group_token_num (int):  Number of global tokens in JCCMTM-E


--​**cross_domain**​ (bool):  Must be set to True for cross-domain scenarios
