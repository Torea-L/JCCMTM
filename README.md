# JCCMTM
# Official implementation of **JCCMTM: Joint Channel-independent and Channel-dependent Strategy for Masked Multivariate Time-Series Modeling**


The experimental procedure consists of three key steps:

[1] ​Pre-training Data Preparation: The processed data is saved as .npy files for reusable storage upon completion;

​[2] Pre-training Phase;

​[3] Sub-task Fine-tuning.

======================================================

[1] Example Bash Command for Pre-training Data Preparation:​

'''bash
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
