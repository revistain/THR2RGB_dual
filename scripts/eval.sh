OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=7 \
python eval.py \
    --test_seq KAIST SNU Valley \
    --datasets_folder /home/jwkim/workspace/dual_THR2RGB/THR2RGB_Cross_Place_Recognition/Datasets \
    --save_dir ./logs \
    --features_dim 768 \
    --foundation_model_path "/home/jwkim/workspace/benchmark_THR2RGB/pretrained/dinov2_vitb14_pretrain.pth" \
    --resume "/home/jwkim/workspace/dual_THR2RGB/THR2RGB_Cross_Place_Recognition/logs/test/260410_122422/best_model.pth"

# Scene 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley']