<!-- 1. Build dev docker container
```bash
cd OpenHands/containers/dev/
bash dev.sh
``` -->
1. create conda env and install packages
```bash
conda create -n openhands-env python=3.12
```
after the env is created and activated
```bash
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
# make sure you are in the OpenHands root directory
make build
```
2. set up env vars
```bash
export SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/all-hands-ai/runtime:0.31-nikolaik
```
3. Run tests
```bash
bash evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.eval_deepseek HEAD CodeActAgent 100 10 1 princeton-nlp/SWE-bench_Verified test
```