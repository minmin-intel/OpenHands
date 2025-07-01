## How to run the benchmark in python env
1. create conda env
```bash
conda create -n openhands-env python=3.12
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
# inside $WORKDIR/OpenHands
make build
# poetry install --all-groups
```
2. set up env vars
```bash
export LOG_ALL_EVENTS=true # very important! need it to be true to get timing logs
export no_proxy=<host-ip-address-of-your-llm-endpoin>
```
3. run benchmark
```bash
# inside $WORKDIR/OpenHands
bash run_swe.sh
```