declare -a ssls=("barlowtwins" "byol" "dino" "moco" "simclr" "swav" "vicreg")
for ssl in "${ssls[@]}"
do
    echo "Running $ssl"
    python main.py --ssl $ssl
done