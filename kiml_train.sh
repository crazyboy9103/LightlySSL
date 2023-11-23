declare -a ssls=("barlowtwins" "byol" "dino" "moco" "simclr" "swav" "vicreg")
num_gpus=2
backbone="resnet50"
dataset="imagenet"
batch_size=256
data_root=/app/input/dataset/$dataset
for ssl in "${ssls[@]}"
do
    echo "Pretraining $ssl"
    python main.py \
        --ssl $ssl \
        --dataset $dataset \
        --data_root $data_root \
        --backbone $backbone \
        --batch_size $batch_size \
        --pretrain_epochs 100 \
        --experiment train \
        --num_gpus $num_gpus
done

eval_epochs=100
backbone_checkpoint=./checkpoints/ssl/$ssl/$backbone/$dataset/ssl_model
for ssl in "${ssls[@]}"
do
    echo "Linear evaluating $ssl"
    python main.py \
        --ssl $ssl \
        --dataset $dataset \
        --data_root $data_root \
        --backbone $backbone \
        --backbone_checkpoint $backbone_checkpoint \
        --batch_size $batch_size \
        --eval_epochs $eval_epochs \
        --experiment eval \
        --sl linear \
        --num_gpus $num_gpus
done

for ssl in "${ssls[@]}"
do
    echo "Finetuning $ssl"
    python main.py \
        --ssl $ssl \
        --dataset $dataset \
        --data_root $data_root \
        --backbone $backbone \
        --backbone_checkpoint $backbone_checkpoint \
        --batch_size $batch_size \
        --eval_epochs $eval_epochs \
        --experiment eval \
        --sl finetune \
        --num_gpus $num_gpus
done