## Requirements:
- python 3.9
- torch 1.10
- torchvision 0.11 
- timm 0.5.4
- transformers 4.11.0


## Training
Below is an example for self-supervised pre-training of a CLIP model on CC3M with 2 GPUs.

```bash
TRANSFORMERS_OFFLINE=1
data_path=/path/to/your/data
train_image_root=cc3m
data=cc3m
train_file=clip_train/${data}_train_new.json
gamma=0.8
rho=8.0

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9800 \
    --use_env clip.py \
    --data_path ${data_path} \
    --train_file ${train_file} \
    --train_image_root ${data} \
    --output_dir output/isogclr_${data}_g${gamma}_rho${rho}_0.01 \
    --init_model \
    --use_amp \
    --ita_type isogclr \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} --rho_init ${rho} \
    --eta_init 0.03 --eta_sched const > ${data}_isogclr_g${gamma}_rho${rho}_0.01.log &
```


## Linear evaluation
After pre-training, we can evaluate the pre-trained models on two downstream tasks: zero-shot retrieval on ms-coco and flickr30k, and zero-shot image classification on CIFAR10, CIFAR100, and ImageNet1k, as follows:
```bash
TRANSFORMERS_OFFLINE=1
data_path=/path/to/your/data
train_image_root=cc3m
data=cc3m
gamma=0.8
rho=8.0
train_file=clip_train/${data}_train_new.json

saved_model=isogclr_${data}_g${gamma}_rho${rho}_0.01

zs_dataset=cifar10

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7800 \
    --use_env clip.py \
    --data_path ${data_path} \
    --train_file ${train_file} \
    --train_image_root ${data} \
    --ita_type clip \
    --output_dir output/evaluate_${saved_model} \
    --evaluate \
    --checkpoint output/${saved_model}/checkpoint_best.pth \
    --use_amp \
    --zs_dataset ${zs_dataset} &

zs_dataset=cifar100

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7900 \
    --use_env clip.py \
    --data_path ${data_path} \
    --train_file ${train_file} \
    --train_image_root ${data} \
    --ita_type clip \
    --output_dir output/evaluate_${saved_model} \
    --evaluate \
    --checkpoint output/${saved_model}/checkpoint_best.pth \
    --use_amp \
    --zs_dataset ${zs_dataset} &

zs_dataset=imagenet

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7700 \
    --use_env clip.py \
    --data_path ${data_path} \
    --train_file ${train_file} \
    --train_image_root ${data} \
    --ita_type clip \
    --output_dir output/evaluate_${saved_model} \
    --evaluate \
    --checkpoint output/${saved_model}/checkpoint_best.pth \
    --zs_datafolder /data/imagenet/imagenet/val \
    --use_amp \
    --zs_dataset ${zs_dataset} &
```




