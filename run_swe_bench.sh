llm_config=llm.eval_deepseek_reasoner
max_iter=50
bash evaluation/benchmarks/swe_bench/scripts/run_infer.sh $llm_config HEAD CodeActAgent 100 $max_iter 1 princeton-nlp/SWE-bench_Verified test