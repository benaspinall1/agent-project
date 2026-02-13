### Dear Grader,

Run this to see the progress on out project

``` bash
python progress.py
```

---
Change these variables to see progress for specific folders/models

```python

if __name__ == "__main__":
  
  model_size = "4B" # 4B, 8B, 14B, 32B
  env = "retail" # retail, airline
  strategy = "act" # act, react, fc
  folder_path = f"results/{env}/{strategy}/{model_size}"
  ...


```

---

### README.md

Check out our **README.md** if you care to see how we are running the experiments.