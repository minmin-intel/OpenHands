MODEL="openai/meta-llama/Llama-3.3-70B-Instruct"
COMMIT_HASH=HEAD
AGENT=CodeActAgent
EVAL_LIMIT=20
MAX_ITER=2
NUM_WORKERS=$EVAL_LIMIT
DATASET="princeton-nlp/SWE-bench_Lite"
SPLIT="test"
N_RUNS=1
MODE="swe"
POISSON_RATE=100.0
BASE_URL="http://10.7.4.57:8086/v1"

./evaluation/benchmarks/swe_bench/scripts/run_infer.sh $MODEL $COMMIT_HASH $AGENT $EVAL_LIMIT $MAX_ITER $NUM_WORKERS $DATASET $SPLIT $N_RUNS $MODE $POISSON_RATE $BASE_URL "true"