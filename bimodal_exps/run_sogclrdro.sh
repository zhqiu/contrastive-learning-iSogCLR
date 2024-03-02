TRANSFORMERS_OFFLINE=1
data_path=/data/qiuzh/ALBEF/data
train_image_root=cc3m
data=cc3m
train_file=clip_train/${data}_train_new.json

python=/home/qiuzh/.conda/envs/vlp/bin/python

gamma=0.8
rho=6.0

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=8800 \
    --use_env clip.py \
    --data_path ${data_path} \
    --train_file ${train_file} \
    --train_image_root ${data} \
    --output_dir output/sogclrdro_${data}_g${gamma}_rho${rho}_0.01 \
    --init_model \
    --use_amp \
    --zs_dataset cifar10 \
    --ita_type sogclr_dro \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} --rho_init ${rho} \
    --eta_init 0.03 --eta_sched const > ${data}_sogclrdro_g${gamma}_rho${rho}_0.01.log &




