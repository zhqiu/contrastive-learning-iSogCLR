## Requirements:
- python 3.9
- torch 1.10
- torchvision 0.11 


## Credits
Our implementation is based on [SupContrast](https://github.com/HobbitLong/SupContrast/).


## Training  
Below is a quick example for self-supervised pre-training of a ResNet-18 model on CIFAR100 with 1 GPU.

```bash
data_folder=./datasets/
data=cifar100
rho=0.4
gamma=0.8
epoch=400

CUDA_VISIBLE_DEVICES=0 python main_supcon.py --batch_size 128 \
    --learning_rate 0.8 \
    --cosine --warm \
    --epochs ${epoch} \
    --model resnet18 \
    --dataset $data \
    --data_folder $data_folder \
    --size 32 \
    --desc dro_${data}_${rho}_${gamma}_${epoch}ep \
    --DRO_mod --DRO_tau_init 0.1 \
    --DRO_gamma ${gamma} --DRO_rho ${rho} --DRO_beta_u 0.5 \
    --DRO_eta_init 0.03  \
    --DRO_eta_sched const  \
    --method SimCLR > isogclr_${data}_${rho}_${gamma}_${epoch}ep.log
```


## Linear evaluation
After pre-training, we can evaluate the pre-trained models as follows:
```bash
data_folder=./datasets/
data=cifar100
rho=0.4
gamma=0.8
epoch=400

CUDA_VISIBLE_DEVICES=0 python main_linear.py --batch_size 128 \
    --model resnet18 \
    --dataset $data \
    --data_folder $data_folder \
    --size 32 \
    --epochs 100 \
    --learning_rate 30.0 \
    --ckpt isogclr_${data}_${rho}_${gamma}_${train_epoch}ep.pth > isogclr_${data}_${rho}_${gamma}_${train_epoch}ep.res
```
