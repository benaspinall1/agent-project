import json
import os
from typing import List

def remove_error_logs(file_path: str) -> None:
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        data = json.load(f)

    # Remove any tasks that have an error recorded
    cleaned_data = [
        item
        for item in data
        if not (item.get("info", {}).get("error") is not None)
    ]

    with open(file_path, "w") as f:
        json.dump(cleaned_data, f, indent=4)
        
def missing_task_ids(file_path: str) -> None:
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        data = json.load(f)
        completed = set([item["task_id"] for item in data])
    path_parts = file_path.replace("\\", "/").split("/")
    env = path_parts[1] if len(path_parts) > 1 else "airline"
    total_tasks = 115 if env == "retail" else 50
    missing = [i for i in range(total_tasks) if i not in completed]
    if len(missing) == 0:
        return
    completed_count = len(completed)
    missing_str = ", ".join(map(str, missing))
    print(f"\n{file_path}")
    print(f"  {completed_count}/{total_tasks} completed | missing: {len(missing)}")


def remove_duplicate_task_ids(file_path: str) -> None:
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        data = json.load(f)

    seen = set()
    cleaned_data = []
    for item in data:
        task_id = item.get("task_id")
        if task_id not in seen:
            cleaned_data.append(item)
            seen.add(task_id)

    with open(file_path, "w") as f:
        json.dump(cleaned_data, f, indent=4)


if __name__ == "__main__":
    base_path = "ben"
    environments = ["airline", "retail"]
    agents = ["act", "react", "fc"]
    models = ["4B", "8B", "14B", "32B"]
    trials = [1, 2, 3, 4, 5]
    for env in environments:
        for agent in agents:
            for model in models:
                for trial in trials:
                    file = f"{base_path}/{env}/{agent}/{model}/num_trials-{trial}.json"
                    remove_error_logs(file)
                    remove_duplicate_task_ids(file)
                    missing_task_ids(file)