# %% [markdown]
# ### Imports and global utils

# %%
'''imports'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,4"
import pickle
# clear GPU memory
from utils import general_utils, eoc
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from typing import Tuple, List
import torch.nn.functional as F
import eoc_datasets
from model_configs import model_configs

# %% [markdown]
# ### Load model

# %%
# Specify model and load tokenizer
model_identifier = "google-ul2"
# model_identifier = "t5-11b" 
# model_identifier = "flan-ul2"

config = model_configs[model_identifier]

model_name, model_dir, mode, no_extra_tokens, kwargs = \
    config['model_name'], config['model_dir'], config['mode'], config['no_extra_tokens'], config['kwargs']

# Use custom huggingface cache dirs in case the default one has low capacity, since the models are large.
MY_HUGGINGFACE_CACHE_DIR ='huggingface_cache'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=os.path.join(MY_HUGGINGFACE_CACHE_DIR, model_dir)
)

# define loss and get extra ids
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) #reduction='avg'
ce_loss_sum = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum') #reduction='sum'

# %%
RUN_CELL = True  # Load model
if RUN_CELL:
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=os.path.join(MY_HUGGINGFACE_CACHE_DIR, model_dir),
        **kwargs
    )

# %%
# dataset_processor = eoc_datasets.ARCProcessor()
# dataset_processor = eoc_datasets.HellaswagProcessor()
# dataset_processor = eoc_datasets.MMLUProcessor()
subjects = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', \
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
    
for subject in subjects:
    print(f"Subject: {subject}")
    dataset_processor = eoc_datasets.MMLUProcessor(subjects=[subject])
      
    data = dataset_processor.get_dataset(
        set_partition='test', 
        # shuffle=True, # for hellaswag; index bias: the first 1000 examples have very low accuracy compared to the whole
        # first_k_instances=1000, # see above 
    )

    example_generator = dataset_processor.example_generator
    RUN_CELL = True   # set tensors_filtering_criterion by lengths
    if RUN_CELL:
        def tensors_filtering_criterion(input_ids, completions_batch):
            # return True
            # remove trailing padding from completions   
            # print('input_ids:', input_ids)
            # print('completions_batch:', completions_batch)
            return len(input_ids[0]) > 20 \
                # and all([len(general_utils.remove_trailing_zeros_from_1d_tensor(completion)) < 6 for completion in completions_batch]) \
                # and not all([len(general_utils.remove_trailing_zeros_from_1d_tensor(completion)) < 4 for completion in completions_batch])
                # and all([len(general_utils.remove_trailing_zeros_from_1d_tensor(completion)) < 5 for completion in completions_batch]) \

        gen = example_generator(data, tokenizer, mode=mode, tensors_filtering_criterion=tensors_filtering_criterion)
        input_lens = []
        completion_lens = []
        for example_id, input_ids, completions_batch, label in tqdm(gen):
            input_lens.append(len(input_ids[0]))
            completion_lens.append(len(completions_batch[0])) # with padding, this is the max len of the completions
        # print(f"input len > 20 and completion len < 10  and len > 6: {sum([i > 20 and j < 6 for i, j in zip(input_lens, completion_lens)])}")
        # print(f"completion len < 6: {sum([j < 6 for j in completion_lens])}")
        print(f"input len max: {max(input_lens)}, min: {min(input_lens)}, avg: {sum(input_lens)/len(input_lens)}")
        print(f"completion len max: {max(completion_lens)}, min: {min(completion_lens)}, avg: {sum(completion_lens)/len(completion_lens)}")
    RUN_CELL = True    # generate baseline info and conditionals
    if RUN_CELL:
        baseline = dict() 
        # save the label and the number of completions
        gen = example_generator(data, tokenizer, mode, tensors_filtering_criterion=tensors_filtering_criterion)
        for example_id, input_ids, completions_batch, label in tqdm(gen):
            baseline[example_id] = dict()
            baseline[example_id]['label'] = label
            baseline[example_id]['no_completions'] = len(completions_batch)
            baseline[example_id]['p_map'] = []
            p_and_completion = []
            outputs = eoc.multi_labels_forward(model, input_ids.cuda(), completions_batch.cuda())

            for completion_index in range(len(completions_batch)):
                p = -ce_loss(
                    # Only care about the tokens corresponding to the last word and omit offset tokens 
                    # if the first one is <extra_id_0> and it is omitted
                    outputs.logits[completion_index][no_extra_tokens:].cuda(), 
                    completions_batch[completion_index][no_extra_tokens:].cuda()
                )

                baseline[example_id]['p_map'] += [p.detach().cpu().tolist()]
    ### K-offset Conditionals
    RUN_CELL = True 
    if RUN_CELL:
        MAX_OFFSET = 10
        p_map_offset = dict() # maps (example_id, offset, completion_index) -> avg_p
        for offset in range(1, MAX_OFFSET+1):
            gen = example_generator(data, tokenizer, mode, tensors_filtering_criterion=tensors_filtering_criterion)
            for example_id, input_ids, completions_batch, label in tqdm(gen):
                input_ids_offset, labels_offset = eoc.create_offset_sample_from_batch(
                    tokenizer,
                    input_ids,
                    completions_batch,
                    offset
                )
                outputs = eoc.multi_labels_forward(model, input_ids_offset.cuda(), labels_offset.cuda())
                for completion_index in range(len(completions_batch)):
                    avg_log_p = -ce_loss(
                        # Only care about the tokens corresponding to the original completion and omit offset tokens 
                        # if the first one is <extra_id_0> and it is omitted
                        outputs.logits[completion_index][no_extra_tokens+offset:].cuda(), 
                        labels_offset[completion_index][no_extra_tokens+offset:].cuda()
                    )
                    p_map_offset[(example_id, offset, completion_index)] = \
                        avg_log_p.detach().cpu().tolist()
    ### Multispan Conditionals
    RUN_CELL = True    # generate multispan conditionals
    if RUN_CELL:
        length_gap_num_tuples = [
            (3, 5, 1),
            (3, 5, 2),
            (3, 3, 1),
            (3, 3, 2),
            (3, 4, 1),
            (3, 4, 2),
            (3, 10, 1),
        ]
        p_map_multispan = dict()
        for length_gap_num_tuple in length_gap_num_tuples:
            span_length, gap_between_spans, num_spans = length_gap_num_tuple    
            gen = example_generator(data, tokenizer, mode, tensors_filtering_criterion=tensors_filtering_criterion)

            for example_id, input_ids, completions_batch, label in tqdm(gen):
                # print(input_ids.shape)
                # continue
                inputs_ids_multispan, labels_multispan = eoc.create_multiple_span_sample_from_batch(
                    tokenizer,
                    input_ids[0], # squeeze 1st dim
                    completions_batch,
                    span_length,
                    gap_between_spans,
                    num_spans,
                )
                outputs = eoc.multi_labels_forward(model, inputs_ids_multispan.cuda(), labels_multispan.cuda())

                for completion_index in range(len(completions_batch)):
                    # assert multispan samples are correct 
                    assert completions_batch[completion_index].nonzero().shape[0] == \
                        labels_multispan[completion_index][num_spans * (span_length + 1) :].nonzero().shape[0]

                    avg_log_p = -ce_loss(
                        # Only care about the tokens corresponding to the completion (see assert below)); 
                        # so the first <extra_id_0> is omitted, and for each span, the span + <extra_id_k> is omitted;
                        # totally 1 + num_spans * (span_length + 1) tokens are omitted;
                        # labels_multispan contains paddings.
                        outputs.logits[completion_index][1 + num_spans * (span_length + 1) :].cuda(), 
                        labels_multispan[completion_index][1 + num_spans * (span_length + 1) :].cuda()
                    )
                    p_map_multispan[(example_id, span_length, gap_between_spans, num_spans, completion_index)] = \
                        avg_log_p.detach().cpu().tolist()
    ### Ensemble of Conditionals
    '''Define the EOC function'''
    # Max reduction to emsemble conditionals for the same last word
    '''Max reduction to emsemble conditionals for the same last word, 
    i.e., only the maximum avg_log_p is kept for each last word across different range_middle_span_length's and range_middle_to_end_gap's.
    Emsemble the baseline conditionals with the K-offset conditionals and middle-off conditionals.'''

    def run_eoc(offsets, length_gap_num_tuples):
        add_baseline = True
        add_k_offset = offsets != []
        add_multispan = length_gap_num_tuples != []

        count_correct = 0
        for example_index in range(len(baseline)):
            no_completions = baseline[example_index]['no_completions']
            # Create a list of tuples (avg_log_p, completion) for each completion
            p_and_completion = []
            
            # add the baseline (offset = 0 from K-offset ensemble) to the list
            if add_baseline:
                p_and_completion += [
                    (baseline[example_index]['p_map'][completion_index], completion_index)
                    for completion_index in range(no_completions)
                ]
                
            # add the whole K-offset ensemble to the list
            if add_k_offset:
                for offset in offsets:
                    p_and_completion += [
                        (p_map_offset[(example_index, offset, completion_index)], completion_index)
                        for completion_index in range(no_completions)
                    ]
                    
            if add_multispan:
                p_and_completion += [
                    (p_map_multispan[(example_index, *length_gap_num, completion_index)], completion_index)
                    for completion_index in range(no_completions)
                    for length_gap_num in length_gap_num_tuples
                ]

            # Find the tuple with the maximum avg_log_p; this is essentially max reduction
            _, best_completion_index = max(p_and_completion, key=lambda x: x[0])
            label = baseline[example_index]['label']
            if (isinstance(label, int) and best_completion_index == label) or \
            (isinstance(label, list) and best_completion_index in label) :# TruthfulQA has multiple correct answers
                count_correct += 1
            
        # print("accuracy:", count_correct / len(baseline))
        return count_correct / len(baseline)
    RUN_CELL = True    # generate multispan conditionals
    if RUN_CELL:
        length_gap_num_tuples = [
            (3, 5, 1),
            (3, 5, 2),
            (3, 3, 1),
            (3, 3, 2),
            (3, 4, 1),
            (3, 4, 2),
            (3, 10, 1),
        ]
        p_map_multispan = dict()
        for length_gap_num_tuple in length_gap_num_tuples:
            span_length, gap_between_spans, num_spans = length_gap_num_tuple    
            gen = example_generator(data, tokenizer, mode, tensors_filtering_criterion=tensors_filtering_criterion)

            for example_id, input_ids, completions_batch, label in tqdm(gen):
                # print(input_ids.shape)
                # continue
                inputs_ids_multispan, labels_multispan = eoc.create_multiple_span_sample_from_batch(
                    tokenizer,
                    input_ids[0], # squeeze 1st dim
                    completions_batch,
                    span_length,
                    gap_between_spans,
                    num_spans,
                )
                outputs = eoc.multi_labels_forward(model, inputs_ids_multispan.cuda(), labels_multispan.cuda())

                for completion_index in range(len(completions_batch)):
                    # assert multispan samples are correct 
                    assert completions_batch[completion_index].nonzero().shape[0] == \
                        labels_multispan[completion_index][num_spans * (span_length + 1) :].nonzero().shape[0]

                    avg_log_p = -ce_loss(
                        # Only care about the tokens corresponding to the completion (see assert below)); 
                        # so the first <extra_id_0> is omitted, and for each span, the span + <extra_id_k> is omitted;
                        # totally 1 + num_spans * (span_length + 1) tokens are omitted;
                        # labels_multispan contains paddings.
                        outputs.logits[completion_index][1 + num_spans * (span_length + 1) :].cuda(), 
                        labels_multispan[completion_index][1 + num_spans * (span_length + 1) :].cuda()
                    )
                    p_map_multispan[(example_id, span_length, gap_between_spans, num_spans, completion_index)] = \
                        avg_log_p.detach().cpu().tolist()
    '''Define the EOC function'''
    # Max reduction to emsemble conditionals for the same last word
    '''Max reduction to emsemble conditionals for the same last word, 
    i.e., only the maximum avg_log_p is kept for each last word across different range_middle_span_length's and range_middle_to_end_gap's.
    Emsemble the baseline conditionals with the K-offset conditionals and middle-off conditionals.'''

    def run_eoc(offsets, length_gap_num_tuples):
        add_baseline = True
        add_k_offset = offsets != []
        add_multispan = length_gap_num_tuples != []

        count_correct = 0
        for example_index in range(len(baseline)):
            no_completions = baseline[example_index]['no_completions']
            # Create a list of tuples (avg_log_p, completion) for each completion
            p_and_completion = []
            
            # add the baseline (offset = 0 from K-offset ensemble) to the list
            if add_baseline:
                p_and_completion += [
                    (baseline[example_index]['p_map'][completion_index], completion_index)
                    for completion_index in range(no_completions)
                ]
                
            # add the whole K-offset ensemble to the list
            if add_k_offset:
                for offset in offsets:
                    p_and_completion += [
                        (p_map_offset[(example_index, offset, completion_index)], completion_index)
                        for completion_index in range(no_completions)
                    ]
                    
            if add_multispan:
                p_and_completion += [
                    (p_map_multispan[(example_index, *length_gap_num, completion_index)], completion_index)
                    for completion_index in range(no_completions)
                    for length_gap_num in length_gap_num_tuples
                ]

            # Find the tuple with the maximum avg_log_p; this is essentially max reduction
            _, best_completion_index = max(p_and_completion, key=lambda x: x[0])
            label = baseline[example_index]['label']
            if (isinstance(label, int) and best_completion_index == label) or \
            (isinstance(label, list) and best_completion_index in label) :# TruthfulQA has multiple correct answers
                count_correct += 1
            
        # print("accuracy:", count_correct / len(baseline))
        return count_correct / len(baseline)
    from itertools import combinations
    import random        
    RUN_CELL = True  # Run EOC
    if RUN_CELL:
        # K-offset conditionals
        ALL_OFFSETS = [1, 2, 3,]
        # Multispan conditionals
        ALL_LENGTH_GAP_NUM_TUPLES = [
            (3, 5, 1),
            (3, 5, 2),
            (3, 3, 1),
            (3, 3, 2),
            (3, 4, 1),
            (3, 4, 2),
        ]
        NO_OFFSETS = len(ALL_OFFSETS)
        NO_MULTISPAN = len(ALL_LENGTH_GAP_NUM_TUPLES)
        NO_DISTS_RANGE = list(range(NO_OFFSETS + NO_MULTISPAN + 1))
        avg_accs = []
        for NO_DISTS in NO_DISTS_RANGE: # no of distributions to ensemble
            all_dist_ids = list(combinations(range(NO_MULTISPAN + NO_OFFSETS), NO_DISTS))
            # shuffle and take the first 100
            random.shuffle(all_dist_ids)
            all_dist_ids = all_dist_ids[:500]
            all_accs = []
            for dist_ids in all_dist_ids:
                offsets = []
                length_gap_num_tuples = []
                for dist_id in dist_ids:
                    if dist_id < NO_OFFSETS:
                        offsets.append(ALL_OFFSETS[dist_id])
                    else:
                        length_gap_num_tuples.append(ALL_LENGTH_GAP_NUM_TUPLES[dist_id - NO_OFFSETS])            
                acc = run_eoc(
                    offsets,
                    length_gap_num_tuples,
                )
                # print offsets and length_gap_num_tuples and acc
                # print(offsets, length_gap_num_tuples, acc)
                all_accs.append(acc)
            avg_acc = sum(all_accs) / len(all_accs)
            avg_accs.append(avg_acc)
            # print number of dists and avg_acc
            print(f"NO_DISTS: {NO_DISTS}, avg_acc: {avg_acc}")



