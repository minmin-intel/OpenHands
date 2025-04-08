output_file=princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_50_N_v0.31.0-no-hint-run_1/output.jsonl
filename=evaluation/evaluation_outputs/outputs/$output_file
instance_id='scikit-learn__scikit-learn-13439 sphinx-doc__sphinx-10435 django__django-11603 django__django-11964 sympy__sympy-15017'

bash evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $filename