import torch

model_configs = {
    "t5-11b": {
        "model_name": "t5-11b",
        "model_dir": "t5-11b",
        "mode": "T5",
        "no_extra_tokens": 1,
        "kwargs": {
            "device_map": "balanced",
            "load_in_8bit": True
        }
    },
    "google-ul2": {
        "model_name": "google/ul2",
        "model_dir": "google-ul2",
        "mode": "[NLG]",
        "no_extra_tokens": 1,
        "kwargs": {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "balanced",
        }
    },
    "flan-ul2": {
        "model_name": "google/flan-ul2",
        "model_dir": "flan-ul2",
        "mode": "Flan-UL2",
        "no_extra_tokens": 0,
        "kwargs": {
            "device_map": "auto",
            "load_in_8bit": True
        }
    }
}

