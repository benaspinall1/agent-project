# Tau-Bench Experiment Execution Workflow

## Jupyter Session Request

Account

```
class_class59827694spring_2026
```

Partition

```
gaudi
```

QOS

```
class_gaudi
```

CPU Core Allocation

```
20
```

Memory Allocation (GiB)

```
40
```

GPU Resources

```
gpu:hl225:1
```

Jupyter Wall Time (your choice)

```
0-#
```

Jupyter lab version

```
latest
```

## Project Folder Structure Overview (In Sol not GitHub)

This repository is organized so that **model servers**, **Slurm job scripts**, and **experiment drivers** are clearly separated. Understanding this layout makes it easier to debug failures and onboard new contributors.

```
~/agent-project/
├── assistant-server.sh        # Launches the assistant vLLM server (OpenAI-compatible)
├── user-server.sh             # Launches the user vLLM server (OpenAI-compatible)
├── user-vllm-job.sh           # Slurm job wrapper for starting the user vLLM server
├── tau-experiment.sh          # Main Slurm job script that runs Tau-Bench
│
├── logs/                      # Slurm stdout logs for servers and experiments
├── errors/                    # Slurm stderr logs for servers and experiments
├── results/                   # Home for the json files
└── tau-bench/                 # Tau-Bench codebase

```

### Key Ideas

- **Server scripts** (`assistant-server.sh`, `user-server.sh`) are responsible only for starting vLLM instances.
- **Slurm job scripts** (`user-vllm-job.sh`, `tau-experiment.sh`) handle resource allocation and scheduling.
- **Tau-Bench itself** lives in its own subdirectory and is treated as an executable dependency.
- **Logs** are the first place to look when something fails.

This separation keeps experiments reproducible and prevents model server issues from contaminating benchmark logic.

---

This document describes the **end-to-end command sequence** required to run a Tau-Bench experiment on the cluster. It is intended to help teammates quickly understand the setup, dependencies, and execution order.


## 1. Project Directory & Script Permissions

First, move into the project directory and ensure all relevant scripts are executable:

```bash
cd ~/agent-project
chmod +x assistant-server.sh user-server.sh tau-experiment.sh user-vllm-job.sh
```

This guarantees that the Slurm job scripts and server startup scripts can be executed without permission errors.

---

## 2. Load Environment & Activate Conda/Mamba Env

Load the cluster-provided mamba module and activate the environment that contains PyTorch + vLLM support:

```bash
module load mamba/latest
source activate gaudi-pytorch-vllm
```

> ⚠️ All subsequent commands assume this environment is active.

---

## 3. Python Dependency Fixes

Some Tau-Bench and LiteLLM dependencies need to be explicitly installed at the user level:

```bash
pip install --user "litellm==1.41.0"
pip install --user "typing_extensions>=4.10.0"
pip install --user "tokenizers>=0.21,<0.22"
```

These versions are known to be compatible with the current Tau-Bench setup and OpenAI-compatible vLLM servers.

---

## 4. Start the User vLLM Server

Submit the Slurm job that launches the **user model** vLLM server:

```bash
sbatch user-vllm-job.sh
```

![Commad output](photos/job-id.png.jpg)

After submission, monitor the logs to confirm the server is running:

```bash
tail -n 20 logs/user_vllm_<jobid>.out
```

You should see some output that looks like the snippet below. If not, go into the **logs** directory and
scroll to the top of the file with this filename structure: "user_vllm_jobid.out"

```
====================================
Starting USER vLLM server on node: gaudi001
User model: Qwen/Qwen3-32B
User server base: http://gaudi001:8007/v1
====================================
...
```

Once running, verify that the OpenAI-compatible API endpoint is live:

```bash
curl -s http://gaudi001:8007/v1/models | jq
```

> ⚠️ IMPORTANT: Dont just copy and paste this command withouth making sure it matches the **User server base** from the **user_vllm_jobid.out**

You should see a JSON response listing the specs of the user model.

---

## 5. Export User Model API Base

Set the environment variable that Tau-Bench uses to locate the **user model** endpoint:

```bash
export USER_MODEL_API_BASE="http://gaudi001:8007/v1"
```

This must be exported **before** launching Tau-Bench experiments.

---

## 6. Launch Tau-Bench Experiments

### Option A: Slurm Job Array (Recommended for Sweeps)

Run using a job array (This command does not assume that you start at 0 and end at 119. Choose any sub array below):

```bash
sbatch --array=0-119 tau-experiment.sh

Copy any line below to run that experiment (same order as the table above):

sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 1    # 0
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 2    # 1
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 3    # 2
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 4    # 3
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 5    # 4
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 1    # 5
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 2    # 6
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 3    # 7
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 4    # 8
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 5    # 9
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 1   # 10
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 2   # 11
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 3   # 12
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 4   # 13
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 5   # 14
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 1   # 15
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 2   # 16
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 3   # 17
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 4   # 18
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 5   # 19
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 1  # 20
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 2  # 21
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 3  # 22
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 4  # 23
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 5  # 24
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 1  # 25
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 2  # 26
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 3  # 27
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 4  # 28
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 5  # 29
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 1 # 30
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 2 # 31
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 3 # 32
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 4 # 33
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 5 # 34
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 1 # 35
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 2 # 36
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 3 # 37
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 4 # 38
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 5 # 39
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 1     # 40
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 2     # 41
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 3     # 42
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 4     # 43
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 5     # 44
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 1     # 45
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 2     # 46
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 3     # 47
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 4     # 48
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 5     # 49
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 1    # 50
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 2    # 51
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 3    # 52
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 4    # 53
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 5    # 54
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 1    # 55
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 2    # 56
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 3    # 57
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 4    # 58
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 5    # 59
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 1     # 60
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 2     # 61
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 3     # 62
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 4     # 63
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 5     # 64
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 1     # 65
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 2     # 66
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 3     # 67
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 4     # 68
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 5     # 69
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 1    # 70
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 2    # 71
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 3    # 72
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 4    # 73
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 5    # 74
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 1    # 75
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 2    # 76
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 3    # 77
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 4    # 78
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 5    # 79
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 1   # 80
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 2   # 81
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 3   # 82
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 4   # 83
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 5   # 84
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 1   # 85
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 2   # 86
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 3   # 87
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 4   # 88
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 5   # 89
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 1  # 90
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 2  # 91
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 3  # 92
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 4  # 93
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 5  # 94
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 1  # 95
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 2  # 96
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 3  # 97
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 4  # 98
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 5  # 99
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 1      # 100
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 2      # 101
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 3      # 102
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 4      # 103
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 5      # 104
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 1      # 105
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 2      # 106
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 3      # 107
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 4      # 108
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 5      # 109
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 1     # 110
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 2     # 111
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 3     # 112
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 4     # 113
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 5     # 114
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 1     # 115
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 2     # 116
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 3     # 117
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 4     # 118
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 5     # 119
```

---

### Option B: Single Experiment Run

Run a single Tau-Bench experiment with explicit arguments:

```bash
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 1
```

test
**Arguments (in order):**

1. Environment (e.g., `retail`)
2. Agent strategy (e.g., `react`)
3. Assistant model ID
4. Trial index or run ID

---

## 7. Summary Flow

**High-level order of operations:**

1. Set permissions on scripts
2. Load mamba + activate environment
3. Install Python dependencies
4. Launch user vLLM server
5. Verify API endpoint
6. Export `USER_MODEL_API_BASE`
7. Submit Tau-Bench jobs

---

If anything fails, the most common issues are:

- Environment not activated
- vLLM server not fully started before Tau-Bench runs
- Missing or mismatched dependency versions
- Incorrect hostname or port in `USER_MODEL_API_BASE`

This document should be kept in sync with any future changes to scripts or dependency versions.
