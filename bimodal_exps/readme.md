## Requirements:
- python 3.9
- torch 1.10
- torchvision 0.11 
- timm 0.5.4
- transformers 4.11.0


## Credits
Our implementation is based on [ALBEF](https://github.com/salesforce/ALBEF).


## Data
Our bimodal experiments need three image-text datasets: CC3M ([download](https://ai.google.com/research/ConceptualCaptions/download)), MS-COCO ([download](https://cocodataset.org/#download)), and Flickr30K ([download](https://shannon.cs.illinois.edu/DenotationGraph/)). Besides, one also needs the following files to build pytorch datasets,  [clip_train](https://drive.google.com/drive/folders/1hAd0956xIztfwq0WrWLTGBx8sNuye595?usp=sharing) and [downstream](https://drive.google.com/drive/folders/1hAd0956xIztfwq0WrWLTGBx8sNuye595?usp=sharing), for pretraining and downstream tasks, respectively.

After downloading the data, one needs to organize the data folder as follows:
```
.
+--cc3m
|  +--cc3m_train
|  +--cc3m_valid
|
+--coco
|  +--train2014
|  +--val2014
|  +--test2015
|
+--flickr30k
|  +--flickr30k_images
|
+--clip_train 
|
+--downstream
```


## Training
```bash
TRANSFORMERS_OFFLINE=1
data_path=/data1/VLP             
imagenet_val_path=/data1/imagenet/val
train_image_root=cc3m
data=cc3m
train_file=clip_train/${data}_train_new.json
lr=8e-4
frac=1.0
desc=isogclr_tempnet
gamma=0.8
rho=7.0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4820 \
    --use_env clip.py \
    --data_path ${data_path} \
    --data ${data} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/isogclr_tempnet_${data}_gamma${gamma}_rho${rho}_${desc} \
    --init_model \
    --use_amp \
    --epochs 30 --lr ${lr} \
    --lr_temp_net 3e-5 \
    --rho ${rho} \
    --train_frac ${frac} \
    --zs_dataset imagenet \
    --zs_datafolder ${imagenet_val_path} \
    --ita_type isogclr_tempnet \
    --sogclr_gamma ${gamma} > isogclr_tempnet_${data}_gamma${gamma}_rho${rho}_${desc}.log &
```


## Evaluation
After pre-training, we can evaluate the pre-trained models on two downstream tasks: zero-shot retrieval on ms-coco and flickr30k, and zero-shot image classification on CIFAR10, CIFAR100, and ImageNet1k, as follows:
```bash
TRANSFORMERS_OFFLINE=1
data_path=/path/to/your/data
train_image_root=cc3m
data=cc3m
gamma=0.8
rho=8.0
train_file=clip_train/${data}_train_new.json
desc=isogclr_tempnet
saved_model=isogclr_tempnet_${data}_gamma${gamma}_rho${rho}_${desc}

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




