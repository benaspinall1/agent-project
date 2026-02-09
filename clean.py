import json
from typing import List
def remove_error_logs(file_path: str) -> None:
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
        
def missing_task_ids(file_path: str) -> List[int]:
    with open(file_path, "r") as f:
        data = json.load(f)
        completed = set([item["task_id"] for item in data])
    return [i for i in range(50) if i not in completed]

if __name__ == "__main__":
    file = "ben/airline/fc/4B/num_trials-1.json"
    remove_error_logs(file)
    print(missing_task_ids(file))