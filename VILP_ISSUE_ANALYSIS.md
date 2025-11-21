# ViLP Implementation Issue Analysis

## The Problem

The ViLP task is failing because `doc_to_visual` and `doc_to_target` are **not being called**. Here's why:

### How lmms-eval Resolves Methods

1. **ConfigurableTask.doc_to_visual()** (lines 1374-1390 in task.py):
   ```python
   def doc_to_visual(self, doc: dict):
       if type(self.config.doc_to_visual) == str:
           # Treats it as field name: doc["image"]
           return [doc[self.config.doc_to_visual]]
       elif callable(self.config.doc_to_visual):
           # Calls the function from YAML
           return self.config.doc_to_visual(doc, ...)
       else:
           # Returns None if not set!
           return self.config.doc_to_visual
   ```

2. **ConfigurableTask.doc_to_target()** (lines 1337-1362):
   ```python
   def doc_to_target(self, doc: dict):
       doc_to_target = self.config.doc_to_target
       if type(doc_to_target) == str:
           # Treats it as field name: doc["answer"]
           return doc[doc_to_target]
       elif callable(doc_to_target):
           # Calls the function from YAML
           return doc_to_target(doc)
       # ... etc
   ```

### The Issue with ViLP

**Current setup:**
- ✅ ViLPTask defines `doc_to_visual()` and `doc_to_target()` as instance methods
- ❌ YAML has NO `doc_to_visual` or `doc_to_target` entries
- ❌ ConfigurableTask's implementation is called, which looks for `self.config.doc_to_visual`
- ❌ Since it's not in YAML, `self.config.doc_to_visual` is `None`
- ❌ Returns `None` instead of calling ViLPTask's methods

**Even though ViLPTask overrides these methods, ConfigurableTask's implementation doesn't call the overridden version - it ONLY looks at `self.config.doc_to_visual` from the YAML!**

## The Solution

You have two options:

### Option 1: Add Functions to YAML (Recommended)

This is what other multi-image tasks do (like llava_interleave_bench).

**Add to vilp.yaml:**
```yaml
doc_to_visual: !function utils.vilp_doc_to_visual
doc_to_target: !function utils.vilp_doc_to_target
```

**Keep in utils.py:**
```python
def vilp_doc_to_visual(doc):
    """Extract the appropriate image based on _image_idx."""
    image_idx = doc.get("_image_idx", 1)
    image_key = f"image{image_idx}"

    if image_key in doc and doc[image_key] is not None:
        return [doc[image_key].convert("RGB")]
    return []

def vilp_doc_to_target(doc):
    """Extract the appropriate answer based on _image_idx."""
    image_idx = doc.get("_image_idx", 1)
    answer_key = f"answer{image_idx}"
    return str(doc.get(answer_key, ""))
```

**Remove from ViLPTask class:**
- Delete the `doc_to_visual` and `doc_to_target` methods from `__init__.py`
- Keep the custom class ONLY for document expansion logic

### Option 2: Set config in __init__ (More Complex)

Modify ViLPTask.__init__ to set the config:

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._expanded_docs_cache = {}

    # Tell config to use our instance methods
    self._config.doc_to_visual = self._instance_doc_to_visual
    self._config.doc_to_target = self._instance_doc_to_target

def _instance_doc_to_visual(self, doc):
    # Implementation here
    ...
```

But this is messier and not the standard pattern.

## What Methods ARE Needed for a Task

### Required in YAML:
1. **`doc_to_visual`** - Extract image(s) from document
   - Can be: string (field name) or !function
   - Must return: list of PIL Images

2. **`doc_to_text`** - Format the text prompt
   - Must be: !function
   - Must return: string

3. **`doc_to_target`** - Extract ground truth answer
   - Can be: string (field name) or !function
   - Must return: string or list

4. **`process_results`** - Process model output
   - Must be: !function
   - Must return: dict of metrics

### Optional in YAML:
5. **`doc_to_choice`** - For multiple choice tasks
6. **`doc_to_messages`** - For multi-turn conversations

### Optional Custom Task Class:
- Override `train_docs()`, `test_docs()`, `validation_docs()` for custom data processing
- Override `_process_docs()` for document transformation (like ViLP's expansion)
- Do NOT override `doc_to_visual` or `doc_to_target` - use YAML instead!

## Summary for ViLP

**The fix:**
1. Move `doc_to_visual` and `doc_to_target` logic from `__init__.py` to `utils.py` as standalone functions
2. Add them to `vilp.yaml`:
   ```yaml
   doc_to_visual: !function utils.vilp_doc_to_visual
   doc_to_target: !function utils.vilp_doc_to_target
   ```
3. Keep the ViLPTask class ONLY for:
   - Document expansion (_expand_doc, _process_docs)
   - Overriding train_docs/test_docs/validation_docs

**Why it's currently failing:**
- ConfigurableTask.doc_to_visual() is being called
- It looks for self.config.doc_to_visual (from YAML)
- Finds None
- Returns None
- Framework can't extract images → fails

**After the fix:**
- ConfigurableTask.doc_to_visual() is called
- It finds self.config.doc_to_visual = vilp_doc_to_visual function
- Calls the function with the expanded doc (which has _image_idx set)
- Function extracts the correct image → works!
