OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=3 \
python train.py \
    --save_dir ./logs \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path=/home/jwkim/workspace/benchmark_THR2RGB/pretrained/dinov2_vitb14_pretrain.pth