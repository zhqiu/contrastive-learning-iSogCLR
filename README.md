# iSogCLR PyTorch Implementation

In this repo, we show how to train a self-supervised model by using [Robust Global Contrastive Loss]() (RGCL) on several unimodal image datasets (e.g., CIFAR10/100, ImageNet100, etc.) and a widely used bimodal image-text dataset [CC3M](https://ai.google.com/research/ConceptualCaptions/download). The code and scripts for reproducing the unimodal and bimodal experimental results in our paper are provided in unimodal_exp and bimodal_exp folder, respectively.

## Reference
If you find this tutorial helpful, please cite our paper:
```
@inproceedings{qiu2023not,
  title={Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization},
  author={Zi-Hao Qiu, Quanqi Hu, Zhuoning Yuan, Denny Zhou, Lijun Zhang, and Tianbao Yang},
  booktitle={International Conference on Machine Learning},
  pages={TBD},
  year={2023},
  organization={PMLR}
}
```
