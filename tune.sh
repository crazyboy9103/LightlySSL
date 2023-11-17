declare -a ssls=("barlowtwins" "byol" "dino" "moco" "simclr" "swav" "vicreg")
for ssl in "${ssls[@]}"
do
    echo "Tuning $ssl"
    python tune.py --ssl $ssl
done