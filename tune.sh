declare -a ssls=("barlowtwins" "byol" "dino" "moco" "simclr" "swav" "vicreg")
for ssl in "${ssls[@]}"
do
    echo "Tuning $ssl"
    python ray_tune.py --ssl $ssl
done