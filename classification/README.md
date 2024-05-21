# Bert for text embedding
CUDA_VISIBLE_DEVICES=2 python main.py --input-size 64 --model-name CmbFormer_S --input-text --use-bert --batch-size 4 --data-path /home/box-train/CMB_Classification/CMB_classification/dataset/CMB --warmup-epochs 30 --epoch 100 --warmup-epochs 30 --output_dir CMB_output-CmbFormer_S

# CLIP for text embedding
CUDA_VISIBLE_DEVICES=2 python main.py --input-size 64 --model-name CmbFormer_S --input-text --batch-size 4 --data-path /home/box-train/CMB_Classification/CMB_classification/dataset/CMB --warmup-epochs 30 --epoch 100 --warmup-epochs 30 --output_dir CMB_output-CmbFormer_S

# BERT for text embedding and clip for image bedding
CUDA_VISIBLE_DEVICES=2 python main.py --input-size 64 --model-name CmbFormer_S --input-text --use-bert --use-clip-image --batch-size 4 --data-path /home/box-train/CMB_Classification/CMB_classification/dataset/CMB --warmup-epochs 30 --epoch 100 --warmup-epochs 30 --output_dir CMB_output-CmbFormer_S

# CLIP for text and image embedding
CUDA_VISIBLE_DEVICES=2 python main.py --input-size 64 --model-name CmbFormer_S --input-text --use-clip-image --batch-size 4 --data-path /home/box-train/CMB_Classification/CMB_classification/dataset/CMB --warmup-epochs 30 --epoch 100 --warmup-epochs 30 --output_dir CMB_output-CmbFormer_S

# prediction
CUDA_VISIBLE_DEVICES=0 python prediction.py --model-name CmbFormer_B --input-text --use-bert --nb-classes 2 --image-path test_images --checkpoint-path CMB_output-CmbFormer_B/last.pth
