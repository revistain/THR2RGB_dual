OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=6 \
python train.py \
    --train_seq Urban \
    --test_seq SNU Valley \
    --datasets_folder /home/jwkim/workspace/dual_THR2RGB/THR2RGB_Cross_Place_Recognition/Datasets \
    --save_dir ./logs \
    --features_dim 768 \
    --comment test \
    --foundation_model_path=/home/jwkim/workspace/benchmark_THR2RGB/pretrained/dinov2_vitb14_pretrain.pth \
    --use_fast_track

# Scene 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley']