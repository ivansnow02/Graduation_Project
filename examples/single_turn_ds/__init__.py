from .interdisciplinary import InterdisciplinaryDataset

# Legacy datasets (math_hard, abg_coqa, medium, bigcodebench) have been archived to
# examples/archive/legacy_datasets/ and are available as reference implementations.
# The main pipeline uses only the 'interdisciplinary' dataset.

datasets_info = {
    "interdisciplinary": {
        "task_desc": "interdisciplinary question generation",
        "class": InterdisciplinaryDataset,
    },
    # To use archived datasets, import from:
    # from examples.archive.legacy_datasets import MATH, AbgCoQA, Medium, BigCodeBench
}
