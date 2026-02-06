## Why `run.py` Needed to Be Changed (Checkpoint Path Bug)

Tau-Bench builds its checkpoint filename dynamically using the agent model,
user model, temperature, and other run metadata. The original code looked
roughly like this:

```python
time_str = datetime.now().strftime("%m%d%H%M%S")
ckpt_path = (
    f"{config.log_dir}/"
    f"{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}"
    f"_range_{config.start_index}-{config.end_index}"
    f"_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
)

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)
```

This assumes that `ckpt_path` is a *flat file path*. That assumption breaks when
the **user model name contains a slash**.

### The Root Cause

In my setup, the user model is:

```
Qwen/Qwen3-32B
```

When this value is interpolated into the filename, the resulting path looks
like:

```
.../tool-calling-Qwen3-4B-Instruct-2507-0.6_range_0--1_user-Qwen/Qwen3-32B-llm_0205125134.json
```

The `/` in `Qwen/Qwen3-32B` is interpreted by the filesystem as a **directory
separator**, not as part of a filename. As a result:

- `user-Qwen/` is treated as a directory
- `Qwen3-32B-llm_0205125134.json` is treated as a file inside that directory

However, the original code only created `config.log_dir`. It did **not** create
the parent directory implied by `ckpt_path`. When Python tried to write the file,
it failed with:

```
FileNotFoundError: [Errno 2] No such file or directory
```

### What the Fix Does

The fix explicitly ensures that **both** of the following exist before writing:

1. The base log directory
2. The full parent directory of `ckpt_path`

```python
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
```

This change makes checkpoint writing robust even when:
- model names include `/`
- user models are remote (e.g. `Qwen/Qwen3-32B`)
- multiple nested result directories are implied by the filename

### Result

After this change:
- Tau-Bench no longer crashes with `FileNotFoundError`
- Results are saved correctly even with hierarchical model IDs
- The code is idempotent and safe across repeated runs

In short, this change was required because **model identifiers are not safe as
flat filenames**, and the filesystem treats `/` as a directory boundary.
