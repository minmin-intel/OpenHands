MODEL="meta-llama/Llama-3.3-70B-Instruct"
MAX_ITER=100 # the max number of agent steps to run
DATASET="SWE-bench_Lite"
SPLIT="test"

poetry run python evaluation/benchmarks/swe_bench/parse_history.py \
    --dataset $DATASET \
    --split $SPLIT \
    --max_iter $MAX_ITER \
    --model $MODEL
