# MODEL="Devstral-Small-2507"  # get the last part of the model name
# MODEL="Llama-3.3-70B-Instruct"  # get the last part of the model name
MODEL="horizon-beta"
MAX_ITER=100
OPENHANDS_VERSION="v0.44.0"
OUTPUT=./evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/${MODEL}_maxiter_${MAX_ITER}_N_${OPENHANDS_VERSION}-no-hint-run_1/output.jsonl
./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh ${OUTPUT}
