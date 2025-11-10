# lmms-eval Workflow Explanation

## How Evaluation Works in lmms-eval

### 1. Task Definition (YAML + utils.py)
Tasks are defined through YAML config + Python utility functions:

```yaml
dataset_path: org/dataset        # HuggingFace dataset
task: "my_task"
test_split: train
output_type: generate_until      # or loglikelihood, multiple_choice

doc_to_visual: !function utils.doc_to_visual    # Extract images
doc_to_text: !function utils.doc_to_text        # Format prompt
doc_to_target: "answer"                         # Ground truth field

process_results: !function utils.process_results  # Per-example processing
metric_list:
  - metric: accuracy
    aggregation: mean              # How to aggregate across dataset
    higher_is_better: true
```

### 2. Data Flow

```
Dataset Load → Document Iteration → Model Inference → Result Processing → Aggregation
```

**Step-by-step:**

1. **Load Dataset**: Framework loads HF dataset using `dataset_path`

2. **For each document**:
   - `doc_to_visual(doc)` → extracts images
   - `doc_to_text(doc)` → formats the text prompt
   - Model generates response

3. **Process Results** (per-example):
   - `process_results(doc, results)` is called for EACH example
   - Takes: original doc + model's prediction(s)
   - Returns: dict with metric values

   Example:
   ```python
   def process_results(doc, results):
       pred = results[0]  # Model's prediction
       gt = doc["ground_truth"]

       return {
           "accuracy": 1.0 if pred == gt else 0.0,
           "other_metric": some_value
       }
   ```

4. **Aggregation** (across all examples):
   - Framework collects all values for each metric
   - Applies aggregation function specified in YAML
   - Built-in: `mean`, `sum`, etc.
   - Custom: `!function utils.my_aggregate_fn`

   Example:
   ```python
   def my_aggregate_fn(results):
       # results is a list of values from process_results
       return sum(results) / len(results)
   ```

### 3. Important Points

- **`process_results` runs once per example** - return per-example metrics
- **Aggregation runs once per metric** - on all collected values
- **Multiple metrics**: Each metric in `metric_list` is tracked separately
- **Nested returns**: For complex aggregation, return dicts with metadata:

  ```python
  return {
      "accuracy": 1.0,
      "by_domain": {"domain": "logos", "correct": True}  # For custom agg
  }
  ```

### 4. Custom Aggregation Pattern

For computing metrics like "accuracy by domain" or "bias ratio":

```python
# In process_results: return metadata
def process_results(doc, results):
    pred = results[0]
    return {
        "accuracy": 1.0 if pred == doc["gt"] else 0.0,
        "domain_acc": {
            "domain": doc["domain"],
            "correct": pred == doc["gt"]
        }
    }

# In aggregation: process all metadata
def aggregate_by_domain(results):
    # results = [{"domain": "logos", "correct": True}, ...]
    domain_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        domain_counts[r["domain"]]["total"] += 1
        if r["correct"]:
            domain_counts[r["domain"]]["correct"] += 1

    return {d: counts["correct"]/counts["total"]
            for d, counts in domain_counts.items()}
```

### 5. Example: Tracking Both Accuracy and Bias Ratio

```python
def process_results(doc, results):
    pred = results[0].strip().lower()
    ground_truth = doc["ground_truth"].strip().lower()
    expected_bias = doc["expected_bias"].strip().lower()

    return {
        "accuracy": 1.0 if pred == ground_truth else 0.0,
        "bias_ratio": 1.0 if pred == expected_bias else 0.0,
    }

# In YAML:
metric_list:
  - metric: accuracy
    aggregation: mean
    higher_is_better: true
  - metric: bias_ratio
    aggregation: mean
    higher_is_better: false  # Lower bias is better!
```

That's it! Both metrics will be tracked and reported separately.
