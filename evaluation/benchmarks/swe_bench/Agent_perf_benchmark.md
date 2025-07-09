## How to run the benchmark in python env on single host machine
1. Set up env vars
```bash
export LOG_ALL_EVENTS=true # very important! need it to be true to get timing logs
export no_proxy=<host-ip-address-of-your-llm-endpoint> # very important if you are in a proxy env
export WORKDIR=<your-work-directory>
```

2. Download this repo.
```bash
cd $WORKDIR
git clone https://github.com/minmin-intel/OpenHands.git
cd OpenHands
git checkout benchmark-dev
```

3. Create conda env.
```bash
conda create -n openhands-env python=3.12
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
# inside $WORKDIR/OpenHands
make build
```

4. Set up directories for storing intermediate outputs.
```bash
cd $WORKDIR
mkdir openhands
mkdir openhands/trajectories
```

5. Prebuild runtime images.

Only change the following variables in the `run_prebuild.sh` script as needed. On single CPU host, limit number of images to build to a few tens. Each image is about 10GB, check your disk space and calculate the number of images that your machine can accomodate before setting the number.
```bash
EVAL_LIMIT=20 # the number of images to prebuild
DATASET="princeton-nlp/SWE-bench_Lite"
SPLIT="test"
```
Then run the script. Note: it may take some time to finish building images.
```bash
# inside $WORKDIR/OpenHands
bash run_prebuild.sh
```
This will randomly sample user-specified number of instances, build the runtime images for those instances in parallel, and save a jsonl file to `$WORKDIR/openhands/` that contains the swe-bench instance_id's and corresponding runtime image names.

6. Launch LLM endpoint.

You can use method of your choice to launch LLM endpoint, for example, using vllm. Only requirement is that this endpoint should be an OpenAI compatible chat-completion endpoint. The LLM endpoint can be launched on a different host machine.


7. Run benchmark.

Before running the command below, open the `run_swe.sh` script and update the variables as needed.
```bash
MODEL="openai/XXX" # example: "openai/meta-llama/Llama-3.3-70B-Instruct". "openai" tells litellm that it is an openai compatible model endpoint. XXX should be the model name that you used to launch your model endpoint.
COMMIT_HASH=HEAD # not to be changed
AGENT=CodeActAgent # not to be changed
EVAL_LIMIT=100 # the number of instances to launch, should be <= # of prebuilt images
MAX_ITER=100 # the max number of agent steps to run, can be set at a large number
NUM_WORKERS=4 # this var is not used when benchmarking with Poisson distribution
DATASET="princeton-nlp/SWE-bench_Lite" # default, can change to other SWE-bench variants. should match that used in run_prebuild.sh
SPLIT="test" # split of the dataset, should match that used in run_prebuild.sh
N_RUNS=1
MODE="swe"
POISSON_RATE=100.0 # number of requests per minute that you want to simulate
BASE_URL="http://${ip_address}/v1" # the model endpoint url, note that you need to end it with /v1
```
Then run the script. Note: it may take some time for the benchmark to finish, especially if you launch many instances. 
```bash
# inside $WORKDIR/OpenHands
bash run_swe.sh
```
This will launch the swe-bench instances following a Poisson time distribution. Logs will be saved in `$WORKDIR/OpenHands/evaluation/evaluation_outputs/`.

8. Parse logs and calculate metrics.
```bash
# inside $WORKDIR/OpenHands
bash run_parse_log.sh
```