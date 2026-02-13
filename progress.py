import os
import json

def run_on_all_files_in_folder(folder_path, function):
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            result = function(os.path.join(folder_path, file))
            if result is not None:  
                print(result)


def print_task_trials(file_path, group_by_task=False):
    """
    For the given file path, print each (task_id, trial) observed in the data.
    If no 'trial' field is present for a task, prints 'None' as its trial.

    If group_by_task is True, prints a hash-map style summary:
        task_id -> [sorted unique list of trials observed]
    """
    import json
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if not group_by_task:
        print(f"\nTask IDs and trial numbers in {file_path}:")
        for item in data:
            task_id = item.get("task_id")
            trial_num = item.get("trial", None)
            print(f"  task_id: {task_id}, trial: {trial_num}")
        return

    # Group/hash-map style: task_id -> list of trials
    task_to_trials = {}
    for item in data:
        task_id = item.get("task_id")
        trial_num = item.get("trial", None)
        if task_id is None:
            continue
        if task_id not in task_to_trials:
            task_to_trials[task_id] = []
        task_to_trials[task_id].append(trial_num)

    print(f"\nTask IDs mapped to trials in {file_path}:")
    for task_id in sorted(task_to_trials.keys()):
        observed = task_to_trials[task_id]
        has_none = any(t is None for t in observed)
        uniques = sorted({t for t in observed if t is not None})
        if has_none:
            trials_list = [None] + uniques
        else:
            trials_list = uniques
        print(f"  {task_id}: {trials_list}")


def missing_task_ids(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
        completed = set([item["task_id"] for item in data])
    path_parts = file_path.replace("\\", "/").split("/")
    env = path_parts[1] if len(path_parts) > 1 else "airline"
    agent = path_parts[2] if len(path_parts) > 2 else "fc"
    model = path_parts[3] if len(path_parts) > 3 else "4B"
    trial_str = (path_parts[-1] if path_parts else "num_trials-1.json").split("-")[-1].split(".")[0]
    try:
        trial = int(trial_str)
    except ValueError:
        trial = 1
    total_tasks = 115 if env == "retail" else 50
    missing = [i for i in range(total_tasks) if i not in completed]
    if len(missing) == 0:
        print(f"\n{file_path}")
        print(f"  All tasks completed")
        return None
    completed_count = len(completed)
    missing_str = ", ".join(map(str, missing))
    print(f"\n{file_path}")
    print(f"  {completed_count}/{total_tasks} completed | missing: {len(missing)} | {missing_str}")
    # print(f"  missing: {missing_str}")

    # Compute sbatch job index (from README mapping) and print the exact command to run
    env_base = 0 if env == "airline" else 60  # airline: 0-59, retail: 60-119
    agent_offset_map = {"act": 0, "react": 20, "fc": 40}
    model_offset_map = {"4B": 0, "8B": 5, "14B": 10, "32B": 15}
    agent_offset = agent_offset_map.get(agent)
    model_offset = model_offset_map.get(model)
    if agent_offset is not None and model_offset is not None and 1 <= trial <= 5:
        job_index = env_base + agent_offset + model_offset + (trial - 1)
        assistant_model = f"Qwen/Qwen3-{model}-Instruct-2507"
        cmd = f"sbatch tau-experiment.sh {env} {agent} {assistant_model} {trial}  # {job_index}"
        print(f"  cmd: {cmd}")


def count_completed_tasks_in_folder(folder_path, total_tasks):
    """
    Count how many completed tasks there are in all .json files in a directory,
    and print the percentage completion for each file.
    """
    print(f"Counting completed tasks in {folder_path}")
    import json
    import os
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not files:
        print("No .json files found in the directory.")
        return

    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Could not load {file}: {e}")
                continue

        completed_tasks = [item for item in data if not item.get("info", {}).get("error")]
        num_completed = len(completed_tasks)

        percent_complete = 100.0 * num_completed / total_tasks
        print(f"{file}: {num_completed}/{total_tasks} tasks completed ({percent_complete:.2f}%)")
    print()
    return num_completed

def progress_by_model(model):
    total_retail_tasks_per_strategy = 575 # 115 * 5 (5 trials per task)
    total_airline_tasks_per_strategy = 250 # 50 * 5 (5 trials per task)
    completion = count_completed_tasks_in_folder(f"results/retail/react/{model}", total_retail_tasks_per_strategy)
    completion += count_completed_tasks_in_folder(f"results/retail/fc/{model}", total_retail_tasks_per_strategy)
    completion += count_completed_tasks_in_folder(f"results/retail/act/{model}", total_retail_tasks_per_strategy)
    
    completion += count_completed_tasks_in_folder(f"results/airline/react/{model}", total_airline_tasks_per_strategy)
    completion += count_completed_tasks_in_folder(f"results/airline/fc/{model}", total_airline_tasks_per_strategy)
    completion += count_completed_tasks_in_folder(f"results/airline/act/{model}", total_airline_tasks_per_strategy)
    completion = completion / ((total_retail_tasks_per_strategy * 3) + (total_airline_tasks_per_strategy * 3)) * 100 # 3 strategies (act, react, fc)
    print(f"Qwen{model} completion: {completion:.2f}%")


def detailed_progress(folder_path):
    print(f"Printing task trials in {folder_path}      -------------------------")
    run_on_all_files_in_folder(folder_path, lambda x: print_task_trials(x, group_by_task=True))
    print(f"Printing missing task ids in {folder_path} -------------------------")
    run_on_all_files_in_folder(folder_path, missing_task_ids)


if __name__ == "__main__":
  
  model_size = "4B" # 4B, 8B, 14B, 32B
  env = "airline" # retail, airline
  strategy = "fc" # act, react, fc
  folder_path = f"results/{env}/{strategy}/{model_size}"
  
#   progress_by_model(model_size)
  detailed_progress(folder_path) # more fined grained view of progress
