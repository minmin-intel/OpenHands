1. Create conda env and install packages
```bash
conda create -n openhands-env python=3.12
```
after the env is created and activated
```bash
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
# make sure you are in the OpenHands root directory
make build
# if you don't want to install pre-commit hooks
# you can run
# make uninstall-hooks
```
2. Set up env vars
```bash
export SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/all-hands-ai/runtime:0.31-nikolaik
```
3. Run benchmarks and get agent outputs
```bash
bash evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.eval_deepseek HEAD CodeActAgent 100 10 1 princeton-nlp/SWE-bench_Verified test
```
4. Score agent outputs using SWE-bench harness
```bash
output_file=path/to/output.jsonl
bash evaluation/benchmarks/swe_bench/scripts/eval_infer.sh evaluation/evaluation_outputs/outputs/$output_file
# example:
# output_file=princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_10_N_v0.31.0-no-hint-run_1/output.jsonl
```
