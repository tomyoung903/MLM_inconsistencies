import torch

# We experiment on bigbench and mmlu subjects where the baseline models are performing decently
model_configs = {
    "t5-11b": {
        "model_name": "t5-11b",
        "model_dir": "t5-11b",
        "mode": "T5",
        "no_extra_tokens": 1,
        "model_kwargs": {
            "device_map": "balanced",
            "load_in_8bit": True
        },
        "lambada_data_path" : "data/pkls/lambada_t5/",
        "bigbench_subjects": [
                'date_understanding',
                'penguins_in_a_table',
                'logical_deduction_five_objects',
                'salient_translation_error_detection',
        ], 
        "mmlu_subjects": ['high_school_european_history', 'clinical_knowledge', 'high_school_government_and_politics', 'high_school_psychology', 'conceptual_physics', 'marketing', 'world_religions', 'computer_security', 'astronomy']
    },
    "google-ul2": {
        "model_name": "google/ul2",
        "model_dir": "google-ul2",
        "mode": "[NLG]",
        "no_extra_tokens": 1,
        "model_kwargs": {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "balanced",
        },
        "lambada_data_path": "data/pkls/lambada_ul2/",
        "bigbench_subjects": [
                'tracking_shuffled_objects_three_objects',
                'logical_deduction_five_objects',
                'logical_deduction_three_objects',
                'disambiguation_qa',
                'date_understanding'
            ],
        "mmlu_subjects": ['high_school_european_history', 'clinical_knowledge', 'high_school_government_and_politics', 'high_school_psychology', 'conceptual_physics', 'marketing', 'world_religions', 'computer_security', 'astronomy']
    },
}
