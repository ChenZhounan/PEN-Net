# PEN-Net
This is a Repository corresponding to ACCV2022 accepted paper ”Complex Handwriting Trajectory Recovery: Evaluation Metrics and Algorithm“. 


## Dataset
The CASIA dataset is available in:

http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

## Training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --cfg configs/sym_attn_z_resdtw_pot.yml
