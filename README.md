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

---

### Preerequisites

Change the file in tau-bench/tau-bench/run.py to this (the explanation for this change can be found in why_run_py_change.md):

```
# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")

    ckpt_path = (
        f"{config.log_dir}/"
        f"{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}"
        f"_range_{config.start_index}-{config.end_index}"
        f"_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
    )

    # Ensure the log directory and the checkpoint directory both exist.
    # This fixes FileNotFoundError when ckpt_path contains slashes from model names,
    # e.g. user model "Qwen/Qwen3-32B" -> "..._user-Qwen/Qwen3-32B-...json"
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                # Directory should already exist from above, but this is idempotent.
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results)

    # Final write of all results
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")

```

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



| Index | ENV     | AGENT | ASSIST_MODEL                 | TRIAL |
| ----- | ------- | ----- | ---------------------------- | ----- |
| 0     | airline | act   | Qwen/Qwen3-4B-Instruct-2507  | 1     |
| 1     | airline | act   | Qwen/Qwen3-4B-Instruct-2507  | 2     |
| 2     | airline | act   | Qwen/Qwen3-4B-Instruct-2507  | 3     |
| 3     | airline | act   | Qwen/Qwen3-4B-Instruct-2507  | 4     |
| 4     | airline | act   | Qwen/Qwen3-4B-Instruct-2507  | 5     |
| 5     | airline | act   | Qwen/Qwen3-8B-Instruct-2507  | 1     |
| 6     | airline | act   | Qwen/Qwen3-8B-Instruct-2507  | 2     |
| 7     | airline | act   | Qwen/Qwen3-8B-Instruct-2507  | 3     |
| 8     | airline | act   | Qwen/Qwen3-8B-Instruct-2507  | 4     |
| 9     | airline | act   | Qwen/Qwen3-8B-Instruct-2507  | 5     |
| 10    | airline | act   | Qwen/Qwen3-14B-Instruct-2507 | 1     |
| 11    | airline | act   | Qwen/Qwen3-14B-Instruct-2507 | 2     |
| 12    | airline | act   | Qwen/Qwen3-14B-Instruct-2507 | 3     |
| 13    | airline | act   | Qwen/Qwen3-14B-Instruct-2507 | 4     |
| 14    | airline | act   | Qwen/Qwen3-14B-Instruct-2507 | 5     |
| 15    | airline | act   | Qwen/Qwen3-32B-Instruct-2507 | 1     |
| 16    | airline | act   | Qwen/Qwen3-32B-Instruct-2507 | 2     |
| 17    | airline | act   | Qwen/Qwen3-32B-Instruct-2507 | 3     |
| 18    | airline | act   | Qwen/Qwen3-32B-Instruct-2507 | 4     |
| 19    | airline | act   | Qwen/Qwen3-32B-Instruct-2507 | 5     |
| 20    | airline | react | Qwen/Qwen3-4B-Instruct-2507  | 1     |
| 21    | airline | react | Qwen/Qwen3-4B-Instruct-2507  | 2     |
| 22    | airline | react | Qwen/Qwen3-4B-Instruct-2507  | 3     |
| 23    | airline | react | Qwen/Qwen3-4B-Instruct-2507  | 4     |
| 24    | airline | react | Qwen/Qwen3-4B-Instruct-2507  | 5     |
| 25    | airline | react | Qwen/Qwen3-8B-Instruct-2507  | 1     |
| 26    | airline | react | Qwen/Qwen3-8B-Instruct-2507  | 2     |
| 27    | airline | react | Qwen/Qwen3-8B-Instruct-2507  | 3     |
| 28    | airline | react | Qwen/Qwen3-8B-Instruct-2507  | 4     |
| 29    | airline | react | Qwen/Qwen3-8B-Instruct-2507  | 5     |
| 30    | airline | react | Qwen/Qwen3-14B-Instruct-2507 | 1     |
| 31    | airline | react | Qwen/Qwen3-14B-Instruct-2507 | 2     |
| 32    | airline | react | Qwen/Qwen3-14B-Instruct-2507 | 3     |
| 33    | airline | react | Qwen/Qwen3-14B-Instruct-2507 | 4     |
| 34    | airline | react | Qwen/Qwen3-14B-Instruct-2507 | 5     |
| 35    | airline | react | Qwen/Qwen3-32B-Instruct-2507 | 1     |
| 36    | airline | react | Qwen/Qwen3-32B-Instruct-2507 | 2     |
| 37    | airline | react | Qwen/Qwen3-32B-Instruct-2507 | 3     |
| 38    | airline | react | Qwen/Qwen3-32B-Instruct-2507 | 4     |
| 39    | airline | react | Qwen/Qwen3-32B-Instruct-2507 | 5     |
| 40    | airline | fc    | Qwen/Qwen3-4B-Instruct-2507  | 1     |
| 41    | airline | fc    | Qwen/Qwen3-4B-Instruct-2507  | 2     |
| 42    | airline | fc    | Qwen/Qwen3-4B-Instruct-2507  | 3     |
| 43    | airline | fc    | Qwen/Qwen3-4B-Instruct-2507  | 4     |
| 44    | airline | fc    | Qwen/Qwen3-4B-Instruct-2507  | 5     |
| 45    | airline | fc    | Qwen/Qwen3-8B-Instruct-2507  | 1     |
| 46    | airline | fc    | Qwen/Qwen3-8B-Instruct-2507  | 2     |
| 47    | airline | fc    | Qwen/Qwen3-8B-Instruct-2507  | 3     |
| 48    | airline | fc    | Qwen/Qwen3-8B-Instruct-2507  | 4     |
| 49    | airline | fc    | Qwen/Qwen3-8B-Instruct-2507  | 5     |
| 50    | airline | fc    | Qwen/Qwen3-14B-Instruct-2507 | 1     |
| 51    | airline | fc    | Qwen/Qwen3-14B-Instruct-2507 | 2     |
| 52    | airline | fc    | Qwen/Qwen3-14B-Instruct-2507 | 3     |
| 53    | airline | fc    | Qwen/Qwen3-14B-Instruct-2507 | 4     |
| 54    | airline | fc    | Qwen/Qwen3-14B-Instruct-2507 | 5     |
| 55    | airline | fc    | Qwen/Qwen3-32B-Instruct-2507 | 1     |
| 56    | airline | fc    | Qwen/Qwen3-32B-Instruct-2507 | 2     |
| 57    | airline | fc    | Qwen/Qwen3-32B-Instruct-2507 | 3     |
| 58    | airline | fc    | Qwen/Qwen3-32B-Instruct-2507 | 4     |
| 59    | airline | fc    | Qwen/Qwen3-32B-Instruct-2507 | 5     |
| 60    | retail  | act   | Qwen/Qwen3-4B-Instruct-2507  | 1     |
| 61    | retail  | act   | Qwen/Qwen3-4B-Instruct-2507  | 2     |
| 62    | retail  | act   | Qwen/Qwen3-4B-Instruct-2507  | 3     |
| 63    | retail  | act   | Qwen/Qwen3-4B-Instruct-2507  | 4     |
| 64    | retail  | act   | Qwen/Qwen3-4B-Instruct-2507  | 5     |
| 65    | retail  | act   | Qwen/Qwen3-8B-Instruct-2507  | 1     |
| 66    | retail  | act   | Qwen/Qwen3-8B-Instruct-2507  | 2     |
| 67    | retail  | act   | Qwen/Qwen3-8B-Instruct-2507  | 3     |
| 68    | retail  | act   | Qwen/Qwen3-8B-Instruct-2507  | 4     |
| 69    | retail  | act   | Qwen/Qwen3-8B-Instruct-2507  | 5     |
| 70    | retail  | act   | Qwen/Qwen3-14B-Instruct-2507 | 1     |
| 71    | retail  | act   | Qwen/Qwen3-14B-Instruct-2507 | 2     |
| 72    | retail  | act   | Qwen/Qwen3-14B-Instruct-2507 | 3     |
| 73    | retail  | act   | Qwen/Qwen3-14B-Instruct-2507 | 4     |
| 74    | retail  | act   | Qwen/Qwen3-14B-Instruct-2507 | 5     |
| 75    | retail  | act   | Qwen/Qwen3-32B-Instruct-2507 | 1     |
| 76    | retail  | act   | Qwen/Qwen3-32B-Instruct-2507 | 2     |
| 77    | retail  | act   | Qwen/Qwen3-32B-Instruct-2507 | 3     |
| 78    | retail  | act   | Qwen/Qwen3-32B-Instruct-2507 | 4     |
| 79    | retail  | act   | Qwen/Qwen3-32B-Instruct-2507 | 5     |
| 80    | retail  | react | Qwen/Qwen3-4B-Instruct-2507  | 1     |
| 81    | retail  | react | Qwen/Qwen3-4B-Instruct-2507  | 2     |
| 82    | retail  | react | Qwen/Qwen3-4B-Instruct-2507  | 3     |
| 83    | retail  | react | Qwen/Qwen3-4B-Instruct-2507  | 4     |
| 84    | retail  | react | Qwen/Qwen3-4B-Instruct-2507  | 5     |
| 85    | retail  | react | Qwen/Qwen3-8B-Instruct-2507  | 1     |
| 86    | retail  | react | Qwen/Qwen3-8B-Instruct-2507  | 2     |
| 87    | retail  | react | Qwen/Qwen3-8B-Instruct-2507  | 3     |
| 88    | retail  | react | Qwen/Qwen3-8B-Instruct-2507  | 4     |
| 89    | retail  | react | Qwen/Qwen3-8B-Instruct-2507  | 5     |
| 90    | retail  | react | Qwen/Qwen3-14B-Instruct-2507 | 1     |
| 91    | retail  | react | Qwen/Qwen3-14B-Instruct-2507 | 2     |
| 92    | retail  | react | Qwen/Qwen3-14B-Instruct-2507 | 3     |
| 93    | retail  | react | Qwen/Qwen3-14B-Instruct-2507 | 4     |
| 94    | retail  | react | Qwen/Qwen3-14B-Instruct-2507 | 5     |
| 95    | retail  | react | Qwen/Qwen3-32B-Instruct-2507 | 1     |
| 96    | retail  | react | Qwen/Qwen3-32B-Instruct-2507 | 2     |
| 97    | retail  | react | Qwen/Qwen3-32B-Instruct-2507 | 3     |
| 98    | retail  | react | Qwen/Qwen3-32B-Instruct-2507 | 4     |
| 99    | retail  | react | Qwen/Qwen3-32B-Instruct-2507 | 5     |
| 100   | retail  | fc    | Qwen/Qwen3-4B-Instruct-2507  | 1     |
| 101   | retail  | fc    | Qwen/Qwen3-4B-Instruct-2507  | 2     |
| 102   | retail  | fc    | Qwen/Qwen3-4B-Instruct-2507  | 3     |
| 103   | retail  | fc    | Qwen/Qwen3-4B-Instruct-2507  | 4     |
| 104   | retail  | fc    | Qwen/Qwen3-4B-Instruct-2507  | 5     |
| 105   | retail  | fc    | Qwen/Qwen3-8B-Instruct-2507  | 1     |
| 106   | retail  | fc    | Qwen/Qwen3-8B-Instruct-2507  | 2     |
| 107   | retail  | fc    | Qwen/Qwen3-8B-Instruct-2507  | 3     |
| 108   | retail  | fc    | Qwen/Qwen3-8B-Instruct-2507  | 4     |
| 109   | retail  | fc    | Qwen/Qwen3-8B-Instruct-2507  | 5     |
| 110   | retail  | fc    | Qwen/Qwen3-14B-Instruct-2507 | 1     |
| 111   | retail  | fc    | Qwen/Qwen3-14B-Instruct-2507 | 2     |
| 112   | retail  | fc    | Qwen/Qwen3-14B-Instruct-2507 | 3     |
| 113   | retail  | fc    | Qwen/Qwen3-14B-Instruct-2507 | 4     |
| 114   | retail  | fc    | Qwen/Qwen3-14B-Instruct-2507 | 5     |
| 115   | retail  | fc    | Qwen/Qwen3-32B-Instruct-2507 | 1     |
| 116   | retail  | fc    | Qwen/Qwen3-32B-Instruct-2507 | 2     |
| 117   | retail  | fc    | Qwen/Qwen3-32B-Instruct-2507 | 3     |
| 118   | retail  | fc    | Qwen/Qwen3-32B-Instruct-2507 | 4     |
| 119   | retail  | fc    | Qwen/Qwen3-32B-Instruct-2507 | 5     |

Copy any line below to run that experiment (same order as the table above):

```bash
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 1   # 0
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 2   # 1
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 3   # 2
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 4   # 3
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 5   # 4
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 1   # 5
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 2   # 6
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 3   # 7
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 4   # 8
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 5   # 9
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 1  # 10
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 2  # 11
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 3  # 12
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 4  # 13
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 5  # 14
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 1  # 15
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 2  # 16
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 3  # 17
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 4  # 18
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 5  # 19
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 1 # 20
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 2 # 21
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 3 # 22
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 4 # 23
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 5 # 24
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 1 # 25
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 2 # 26
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 3 # 27
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 4 # 28
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 5 # 29
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
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 1    # 40
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 2    # 41
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 3    # 42
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 4    # 43
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 5    # 44
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 1    # 45
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 2    # 46
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 3    # 47
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 4    # 48
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 5    # 49
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 1   # 50
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 2   # 51
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 3   # 52
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 4   # 53
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 5   # 54
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 1   # 55
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 2   # 56
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 3   # 57
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 4   # 58
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 5   # 59
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 1    # 60
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 2    # 61
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 3    # 62
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 4    # 63
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 5    # 64
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 1    # 65
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 2    # 66
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 3    # 67
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 4    # 68
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 5    # 69
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 1   # 70
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 2   # 71
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 3   # 72
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 4   # 73
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 5   # 74
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 1   # 75
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 2   # 76
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 3   # 77
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 4   # 78
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 5   # 79
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 1  # 80
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 2  # 81
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 3  # 82
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 4  # 83
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 5  # 84
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 1  # 85
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 2  # 86
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 3  # 87
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 4  # 88
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 5  # 89
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 1 # 90
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 2 # 91
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 3 # 92
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 4 # 93
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 5 # 94
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 1 # 95
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 2 # 96
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 3 # 97
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 4 # 98
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 5 # 99
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 1     # 100
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 2     # 101
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 3     # 102
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 4     # 103
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 5     # 104
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 1     # 105
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 2     # 106
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 3     # 107
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 4     # 108
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 5     # 109
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 1    # 110
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 2    # 111
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 3    # 112
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 4    # 113
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 5    # 114
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 1    # 115
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 2    # 116
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 3    # 117
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 4    # 118
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 5    # 119
```

This is typically used when running many trials or configurations in parallel.

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
