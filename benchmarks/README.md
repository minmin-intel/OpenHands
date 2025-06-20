
## Setup env
1. create conda env
```bash
conda create -n openhands-env python=3.12
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
# inside $WORKDIR/OpenHands
# make build
poetry install --all-groups
```
Environment variables > config.toml variables > default variables

2. If in proxy env, make sure localhost is in your no_proxy