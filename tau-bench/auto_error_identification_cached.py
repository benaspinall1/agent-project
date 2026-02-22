# Copyright Sierra

import json
import argparse
from enum import Enum
from pydantic import BaseModel
from tau_bench.model_utils import default_api_from_args, API
from tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS
from tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS
from tau_bench.model_utils.args import api_parser
from tau_bench.types import Task, Action
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

def get_args() -> argparse.Namespace:
    parser = api_parser()
    parser.add_argument("--env", type=str, required=True, choices=["airline", "retail"], help="The environment that the original trajectories are from (used to fetch the user instructions)")
    parser.add_argument("--results-path", type=str, help="Path to the results file")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of concurrent API calls")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--max-num-failed-results", "-n", type=int, help="Maximum number of failed results to analyze")
    return parser.parse_args()

class OriginalResult(BaseModel):
    task_id: int
    user_instruction: str
    traj: List[Dict[str, Any]]
    ground_truth_actions: List[Action]
    ground_truth_outputs: List[str]

class FaultAuthor(Enum):
    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"

class FaultAssignmentResult(BaseModel):
    task_id: int
    author: FaultAuthor
    description: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "author": self.author.value,
            "description": self.description,
        }

class FaultType(Enum):
    CALLED_WRONG_TOOL = "called_wrong_tool"
    USED_WRONG_TOOL_ARGUMENT = "used_wrong_tool_argument"
    GOAL_PARTIALLY_COMPLETED = "goal_partially_completed"
    OTHER = "other"

class FaultTypeResult(BaseModel):
    task_id: int
    fault_type: FaultType
    description: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fault_type": self.fault_type.value,
            "description": self.description,
        }

class GradingStrategy(Enum):
    ACTIONS = "actions"
    OUTPUTS = "outputs"

def context_description(grading_strategy: GradingStrategy) -> str:
    if grading_strategy == GradingStrategy.ACTIONS:
        return """You will be given a user instruction, the ground truth action sequence, and a trajectory.
- The user instruction is the instruction given to the simulated user.
- The ground truth action sequence is one example of a valid sequence of actions that lead to the goal state (the sequence of actions could be empty, meaning that no action should have been taken).
- The trajectory is the sequence of messages between the user and the agent.
- The trajectory has been determined to have a fault."""
    return """You will be given a user instruction, the set of required agent response outputs, and a trajectory.
- The user instruction is the instruction given to the simulated user.
- The required agent response outputs are the set of outputs that the agent is expected to communicate to the user.
- The trajectory is the sequence of messages between the user and the agent.
- The trajectory has been determined to have a fault."""

def display_traj(traj: List[Dict[str, Any]]) -> str:
    if len(traj) == 0:
        return "(No agent actions recorded; trajectory is empty.)"
    stripped_traj = [item for item in traj if item["role"] != "system"]
    return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in stripped_traj])

def display_actions(actions: List[Action]) -> str:
    return json.dumps([action.model_dump() for action in actions], indent=4)

def display_context(user_instruction: str, ground_truth_actions: List[Action], ground_truth_outputs: List[str], trajectory: List[Dict[str, Any]]) -> str:
    traj_display = display_traj(trajectory)
    context = f"""----- start user instruction -----
{user_instruction}
----- end user instruction -----"""
    if len(ground_truth_outputs) > 0:
        context += f"""

----- start required outputs -----
{ground_truth_outputs}
----- end required outputs -----"""
    else:
        context += f"""

----- start ground truth action sequence -----
{display_actions(ground_truth_actions)}
----- end ground truth action sequence -----

----- start trajectory -----
{traj_display}
----- end trajectory -----\n"""
    return context


def _write_output(
    output_path: str,
    fault_assignment_results: List[FaultAssignmentResult],
    fault_type_results: List[FaultTypeResult],
) -> None:
    """Write current results to disk. Called after each trajectory so progress is never lost."""
    with open(output_path, "w") as f:
        json.dump(
            {
                "fault_assignment_analysis": [r.model_dump() for r in fault_assignment_results],
                "fault_type_analysis": [r.model_dump() for r in fault_type_results],
            },
            f,
            indent=4,
        )


def _assign_fault_one(
    api: API,
    task_id: int,
    user_instruction: str,
    traj: List[Dict[str, Any]],
    ground_truth_actions: List[Action],
    ground_truth_outputs: List[str],
) -> FaultAssignmentResult:
    """Run fault assignment for a single trajectory. May raise on LLM/API errors."""
    idx_to_author = {
        0: FaultAuthor.USER,
        1: FaultAuthor.AGENT,
        2: FaultAuthor.ENVIRONMENT,
    }
    grading_strategy = GradingStrategy.OUTPUTS if len(ground_truth_outputs) > 0 else GradingStrategy.ACTIONS
    ctx_desc = context_description(grading_strategy)
    context = display_context(user_instruction, ground_truth_actions, ground_truth_outputs, traj)
    res = api.classify(
        instruction=f"{ctx_desc}\n\nDetermine the entity that is responsible for the fault. The user is responsible for the fault if they caused an action that was not grounded in the user instruction. The agent is responsible for the fault if they took an action that was not correct (or took the action with the wrong arguments). The environment is responsible for all other faults.",
        text=context,
        options=["The user", "The agent", "The environment (neither user nor agent)"],
    )
    author = idx_to_author[res]
    description = api.generate(
        instruction=f"{ctx_desc}\n\nDescribe the reason why {author.value} is responsible for the fault in the trajectory. Be concise and only focus on the functional differences between the ground truth and the trajectory.",
        text=context,
    )
    return FaultAssignmentResult(task_id=task_id, author=author, description=description)


def _get_fault_type_one(
    api: API,
    task_id: int,
    user_instruction: str,
    traj: List[Dict[str, Any]],
    ground_truth_actions: List[Action],
    ground_truth_outputs: List[str],
) -> FaultTypeResult:
    """Run fault type classification for a single trajectory. May raise on LLM/API errors."""
    idx_to_fault_type = {
        0: FaultType.CALLED_WRONG_TOOL,
        1: FaultType.USED_WRONG_TOOL_ARGUMENT,
        2: FaultType.GOAL_PARTIALLY_COMPLETED,
        3: FaultType.OTHER,
    }
    grading_strategy = GradingStrategy.OUTPUTS if len(ground_truth_outputs) > 0 else GradingStrategy.ACTIONS
    ctx_desc = context_description(grading_strategy)
    context = display_context(user_instruction, ground_truth_actions, ground_truth_outputs, traj)
    res = api.classify(
        instruction=f"{ctx_desc}\n\nDetermine the type of fault of the first instance of the fault.",
        text=context,
        options=["The user called the wrong tool", "The user used the correct tool with a wrong argument", "The goal was only partially completed", "Other"],
    )
    fault_type = idx_to_fault_type[res]
    description = api.generate(
        instruction=f"{ctx_desc}\n\nDescribe the reason why the following trajectory contains a fault of type \"{fault_type.value}\". Be concise and only focus on the functional differences between the ground truth and the trajectory.",
        text=context,
    )
    return FaultTypeResult(task_id=task_id, fault_type=fault_type, description=description)

def main() -> None:
    args = get_args()
    api = default_api_from_args(args)
    with open(args.results_path, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results")
    env = args.env
    if env == "airline":
        tasks: List[Task] = AIRLINE_TASKS
    elif env == "retail":
        tasks: List[Task] = RETAIL_TASKS
    else:
        raise ValueError(f"Invalid environment: {env}")
    failed_results = [r for r in results if r["reward"] <= 1e-3]
    print(f"Found {len(failed_results)} failed trajectories")
    if args.max_num_failed_results is not None and len(failed_results) > args.max_num_failed_results:
        print(f"Limiting to {args.max_num_failed_results} failed trajectories")
        failed_results = failed_results[:args.max_num_failed_results]
    original_results = []
    for result in failed_results:
        task_id: int = result["task_id"]
        task = tasks[task_id]
        user_instruction = task.instruction
        ground_truth_actions = task.actions
        ground_truth_outputs = task.outputs
        original_result = OriginalResult(task_id=task_id, user_instruction=user_instruction, traj=result["traj"], ground_truth_actions=ground_truth_actions, ground_truth_outputs=ground_truth_outputs)
        original_results.append(original_result)
    # Phase 1: fault assignment — one trajectory at a time, write after each (so crashes don't lose progress)
    fault_assignment_results: List[FaultAssignmentResult] = []
    fault_type_results: List[FaultTypeResult] = []
    print(f"Performing fault assignment on {len(original_results)} failed trajectories (writing after each)...")
    for i, orig in enumerate(original_results):
        print(f"  [{i + 1}/{len(original_results)}] task_id={orig.task_id} ...", end=" ", flush=True)
        try:
            r = _assign_fault_one(
                api, orig.task_id, orig.user_instruction, orig.traj,
                orig.ground_truth_actions, orig.ground_truth_outputs,
            )
            fault_assignment_results.append(r)
            print(r.author.value, flush=True)
        except Exception as e:
            fault_assignment_results.append(
                FaultAssignmentResult(
                    task_id=orig.task_id,
                    author=FaultAuthor.ENVIRONMENT,
                    description=f"Analysis failed: {e!s}",
                )
            )
            print(f"FAILED ({e!s})", flush=True)
        _write_output(args.output_path, fault_assignment_results, fault_type_results)

    failures_due_to_agent = [original_results[i] for i, r in enumerate(fault_assignment_results) if r.author == FaultAuthor.AGENT]
    print(f"Performing fault type analysis on {len(failures_due_to_agent)} agent-caused failures (writing after each)...")
    for i, orig in enumerate(failures_due_to_agent):
        print(f"  [{i + 1}/{len(failures_due_to_agent)}] task_id={orig.task_id} ...", end=" ", flush=True)
        try:
            r = _get_fault_type_one(
                api, orig.task_id, orig.user_instruction, orig.traj,
                orig.ground_truth_actions, orig.ground_truth_outputs,
            )
            fault_type_results.append(r)
            print(r.fault_type.value, flush=True)
        except Exception as e:
            fault_type_results.append(
                FaultTypeResult(
                    task_id=orig.task_id,
                    fault_type=FaultType.OTHER,
                    description=f"Analysis failed: {e!s}",
                )
            )
            print(f"FAILED ({e!s})", flush=True)
        _write_output(args.output_path, fault_assignment_results, fault_type_results)

    n = len(fault_assignment_results)
    if n > 0:
        n_user = sum(1 for r in fault_assignment_results if r.author == FaultAuthor.USER)
        n_agent = sum(1 for r in fault_assignment_results if r.author == FaultAuthor.AGENT)
        n_env = sum(1 for r in fault_assignment_results if r.author == FaultAuthor.ENVIRONMENT)
        print(f"""Reviewed {n} trajectories:

Author fault distribution:
  - User: {n_user} ({round(n_user / n * 100, 2)}%)
  - Agent: {n_agent} ({round(n_agent / n * 100, 2)}%)
  - Environment (otherwise case): {n_env} ({round(n_env / n * 100, 2)}%)
""")
    if fault_type_results:
        nt = len(fault_type_results)
        print("Fault type distribution (only failures marked as being caused by the agent):")
        for ft in FaultType:
            cnt = sum(1 for r in fault_type_results if r.fault_type == ft)
            print(f"  - {ft.value}: {cnt} ({round(cnt / nt * 100, 2)}%)")
    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    main()
