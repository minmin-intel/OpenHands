output_file=princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-reasoner_maxiter_20_N_v0.31.0-no-hint-run_1/output.jsonl
filename=evaluation/evaluation_outputs/outputs/$output_file

bash evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $filename