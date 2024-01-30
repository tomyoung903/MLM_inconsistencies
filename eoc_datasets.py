from typing import Tuple, Callable, Union, List
from datasets import load_dataset, concatenate_datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

def tokenize_input_and_completions(input_: str, completions: list[str], tokenizer: AutoTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize input and completions with a given tokenizer.
    
    Args:
    - input_ (str): The input string.
    - completions (list[str]): List of completion strings.
    - tokenizer (PreTrainedTokenizer): The tokenizer to use.
    
    Returns:
    - Tuple containing input_ids (1 * len) and completions_ids_padded (no_completionss * len).
    """
    input_ids = tokenizer(input_, return_tensors="pt").input_ids
    completions_ids = [tokenizer(completion, return_tensors="pt").input_ids[0, :-1] for completion in completions] # squeeze 1st dim and remove <eos> token with [0,:-1]
    completions_ids_padded = pad_sequence(completions_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids, completions_ids_padded


class DatasetProcessor:
    """
    A class to process different datasets with similar structure.
    """
    def __init__(self, dataset_path: str, subset: Union[str, List] = None):
        self.dataset_path = dataset_path
        self.subset = subset

    def get_dataset(self, set_partition: str, shuffle: bool = False, first_k_instances: int = None, seed: int = 42):
        """
        Load and optionally shuffle and truncate the dataset.
        """
        # if subset is str
        if isinstance(self.subset, str):
            data = load_dataset(self.dataset_path, self.subset)[set_partition]
        elif self is None:
            data = load_dataset(self.dataset_path)[set_partition]
        elif isinstance(self.subset, list):
            all_datasets = [load_dataset(self.dataset_path, subset)[set_partition] for subset in self.subset]
            data = concatenate_datasets(all_datasets)

        if shuffle:
            data = data.shuffle(seed)
        if first_k_instances is not None:
            data = data.select(range(first_k_instances))
        return data

    def example_generator(self, docs, tokenizer, ul2_mode='[NLG]', tensors_filtering_criterion=None) -> Tuple:
        """
        Generate examples from the dataset.
        """
        example_id = 0
        for doc in docs:
            input_, completions = self.prepare_input_and_completions(doc, ul2_mode)
            input_ids, completions_ids_padded = tokenize_input_and_completions(input_, completions, tokenizer)
            
            if tensors_filtering_criterion and not tensors_filtering_criterion(input_ids, completions_ids_padded):
                continue
            
            yield example_id, input_ids, completions_ids_padded, self.get_ground_truth_index(doc)
            example_id += 1

    def prepare_input_and_completions(self, doc, ul2_mode: str) -> Tuple[str, list]:
        """
        Prepare input and completions based on the dataset and UL2 mode.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_ground_truth_index(self, doc) -> int:
        """
        Get the index of the ground truth completion.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class HellaswagProcessor(DatasetProcessor):
    def __init__(self):
        super().__init__(dataset_path="Rowan/hellaswag")

    def prepare_input_and_completions(self, doc, ul2_mode: str) -> Tuple[str, list]:
        endings_list = doc['endings']
        input_ = f"{ul2_mode} {doc['activity_label']}: {doc['ctx']} <extra_id_0>"
        completions = [f"<extra_id_0> {ending}" if ul2_mode == "[NLG]" else f"{ending}" for ending in endings_list]
        return input_, completions

    def get_ground_truth_index(self, doc) -> int:
        return int(doc['label'])


class ARCProcessor(DatasetProcessor):
    def __init__(self):
        # for ai2_arc, subset can be "ARC-Easy", "ARC-Challenge"
        super().__init__(dataset_path="ai2_arc", subset="ARC-Challenge")

    def prepare_input_and_completions(self, doc, ul2_mode: str) -> Tuple[str, list]:
        """
        Prepare input and completions specific to the ARC dataset.
        """
        texts = doc['choices']['text']
        input_ = f"{ul2_mode} Question: {doc['question']} <extra_id_0>"
        if ul2_mode == "[NLG]":
            completions = [f"<extra_id_0> Answer: {text}" for text in texts]
        elif ul2_mode == "[S2S]":
            completions = [f"Answer: {text}" for text in texts]
        else:
            raise ValueError("ul2_mode not defined")
        return input_, completions

    def get_ground_truth_index(self, doc) -> int:
        """
        Get the index of the ground truth answer for the ARC dataset.
        """
        answer_key = doc['answerKey']
        return doc['choices']['label'].index(answer_key)


class MMLUProcessor(DatasetProcessor):
    def __init__(self):
        SUBJECTS = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', \
            'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', \
            'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', \
            'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', \
            'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', \
            'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', \
            'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', \
            'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', \
            'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', \
            'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', \
            'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']
        super().__init__(dataset_path="lukaemon/mmlu", subset=SUBJECTS)

    def prepare_input_and_completions(self, doc, ul2_mode: str) -> Tuple[str, list]:
        """
        Prepare input and completions specific to the MMLU dataset.
        """
        keys = ["A", "B", "C", "D"]
        input_ = ul2_mode + " " + doc['input'] + " " + "<extra_id_0>"
        completions = [f"<extra_id_0> {doc[key]}" for key in keys]
        return input_, completions

    def get_ground_truth_index(self, doc) -> int:
        """
        Get the index of the ground truth answer for the MMLU dataset.
        """
        key_to_index = {"A":0, "B":1, "C":2, "D":3}
        return key_to_index[doc['target']]
