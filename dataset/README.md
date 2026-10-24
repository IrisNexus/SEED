# Dataset Layout

The repository provides model-specific dataset folders:

```text
dataset/
  qwen7b/
  qwen38/
  llama/
```

Each folder contains the same benchmark names:

```text
aim.json
drattack.json
gcg.json
hcot.json
pair.json
renellm.json
```

Sample counts in **each** model folder:

| Dataset | Samples |
| --- | ---: |
| AIM | 100 |
| DRattack | 100 |
| GCG | 100 |
| H-CoT | 50 |
| PAIR | 100 |
| ReNeLLM | 100 |

`max_samples` is an upper limit. The runner uses at most the number of rows
available in the selected file; it does not repeat samples to reach the limit.
For H-CoT, use `50` to process all available samples.

Filenames are consistent across model folders, but file contents can differ.
Select the folder for the model being tested, for example:

```bash
bash scripts/run_sglang.sh qwen7b/gcg.json false 100 true
bash scripts/run_sglang.sh qwen38/gcg.json false 100 true
bash scripts/run_sglang.sh llama/gcg.json false 100 true
```
