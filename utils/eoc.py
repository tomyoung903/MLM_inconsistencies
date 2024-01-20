from typing import List, Dict, Any, Iterable, Union, Tuple # for type hinting
import torch
from tqdm import tqdm



def calculate_num_spans(input_ids: torch.Tensor, span_length: int, gap_between_spans: int, auto_ratio: float = 0.3, max_num_spans = 99):
    '''Calculate the number of spans for the given example_id, span_length, and gap_between_spans. Minimum 1.'''
    considered_length = len(input_ids) - 12 # excluding extra_id_0, eos_token_id, sentinel token, and the first few (~8) tokens
    num_spans = int(auto_ratio * considered_length // (span_length + gap_between_spans))
    num_spans = min(num_spans, max_num_spans)
    num_spans = max(num_spans, 1)
    return num_spans


def create_multiple_span_sample(tokenizer,
                                inputs: str,
                                labels: torch.Tensor, # 1D
                                span_length: int = 5,
                                gap_between_spans: int = 5,
                                num_spans: int = 2,
                                return_tensor: str = "inputs"):
    '''
    This function is a generalization of create_middle_off_sample. It manipulates the input sequence by replacing specified spans with unique <extra_id_n> tokens for each span.
    It handles any number of spans defined by 'num_spans'. Each span of length 'span_length' is separated by a gap of 'gap_between_spans' tokens.

    A num_spans=2 example:

        input_ids (1*len) == [input_regular_tokens, extra_id_0, eos_token_id]

        (break down input_regular_tokens) -> 
        input_regular_tokens_keep, input_regular_tokens_move_0, gap_0, input_regular_tokens_move_1, gap_1, extra_id_0, eos_token_id

        (replace input_regular_tokens_move_0 and input_regular_tokens_move_1 with <extra_id_0> and <extra_id_1> respectively) -> 
        input_regular_tokens_keep, extra_id_0, gap_0, extra_id_1, gap_1, extra_id_2, eos_token_id
            
        labels (1D) == [extra_id_0, labels_regular_tokens]
            (add input_regular_tokens_move_0 and input_regular_tokens_move_1 to the front and change <extra_id_0> and <extra_id_1> respectively) ->
            extra_id_0, input_regular_tokens_move_0, extra_id_1, input_regular_tokens_move_1, extra_id_2, labels_regular_tokens
        
    
    The function segments the input_regular_tokens into multiple spans and gaps, replacing each span with a corresponding <extra_id_n>.
    The labels are also adjusted to match the new structure of the input sequence.

    Parameters:
    inputs: str - The input text.
    labels: torch.Tensor - The corresponding labels.
    span_length: int - The length of each span to be replaced.
    gap_between_spans: int - The length of the gap between spans.
    num_spans: int - The number of spans to replace.
    '''

    input_ids = tokenizer(inputs, return_tensors="pt").input_ids[0]
    total_length = len(input_ids) - 2  # excluding extra_id_0 and eos_token_id

    # Calculate the starting point for the first span
    total_span_length = (span_length + gap_between_spans) * num_spans # Total length of all spans and gaps
    start_of_span = total_length - total_span_length

    input_ids_return = input_ids[:start_of_span]
    # initialize labels_return as 1*0 tensor
    labels_return = torch.tensor([],dtype=torch.int64)
    for i in range(num_spans):
        extra_id_token = torch.tensor([tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")])

        input_ids_return = torch.cat((input_ids_return, extra_id_token))
        labels_return = torch.cat((labels_return, extra_id_token))

        # Add the gap between spans to input_ids_return
        gap_start = start_of_span + i * (span_length + gap_between_spans) + span_length
        gap_end = gap_start + gap_between_spans
        input_ids_return = torch.cat((input_ids_return, input_ids[gap_start:gap_end]))

        # Add the span to the labels_return
        span_start = start_of_span + i * (span_length + gap_between_spans)
        span_end = span_start + span_length
        labels_return = torch.cat((labels_return, input_ids[span_start:span_end]))
        
    
    # Add the last extra_id_token and eos_token_id to input_ids_return
    extra_id_token = torch.tensor([tokenizer.convert_tokens_to_ids(f"<extra_id_{num_spans}>")])
    input_ids_return = torch.cat((input_ids_return, extra_id_token, input_ids[-1:]))

    # Add the last extra_id_token and the labels_regular_tokens to labels_return
    labels_return = torch.cat((labels_return, extra_id_token, labels[1:]))

    if return_tensor == "inputs":
        return input_ids_return
    elif return_tensor == "labels":
        return labels_return



def get_multiple_span_samples(self,
                                id_to_completions_ids: Dict[int, List[torch.Tensor]],
                                length_gap_num_tuple: tuple = (3, 5, None),
                                max_num_spans = 99, # <extra_id_k> goes up to <extra_id_99>
                                auto_ratio = 0.3, # if num_spans is None, then the no. spans decided by input length is scaled by this ratio
                                to_gpu=False):
    '''Apply create_multiple_span_sample to all the completions of each example with length_gap_num_tuple.
    If num_spans is None, then the number of spans is decided by the length of the input.'''
    dataset_multiple_spans = {}
    span_length, gap_between_spans, num_spans = length_gap_num_tuple
    is_num_spans_given = num_spans != None

    for example_id in tqdm(range(len(id_to_completions_ids))):
        # skip if there is no completion
        if len(id_to_completions_ids[example_id]) == 0:
            continue
        
        if not is_num_spans_given: # decide via length of input for each example
            num_spans = self.calculate_num_spans(example_id, span_length, gap_between_spans, auto_ratio, max_num_spans)

        inputs = self.create_multiple_span_sample(
            self.dataset[example_id]['inputs_pretokenized'],
            id_to_completions_ids[example_id][0], # any completion is fine to get the input_ids
            span_length=span_length,
            gap_between_spans=gap_between_spans,
            num_spans=num_spans,
            return_tensor="inputs"
        )

        # get the list of labels first then pad them to the same length for a large tensor
        labels = [
            self.create_multiple_span_sample(
                self.dataset[example_id]['inputs_pretokenized'],
                completion,
                span_length=span_length,
                gap_between_spans=gap_between_spans,
                num_spans=num_spans,
                return_tensor="labels"
            )
            for completion in id_to_completions_ids[example_id]
        ]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        if to_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        if is_num_spans_given:
            dataset_multiple_spans[(example_id, span_length, gap_between_spans, num_spans)] = {
                    "inputs": inputs,
                    "labels": labels
                }
        else:
            dataset_multiple_spans[(example_id, span_length, gap_between_spans, auto_ratio)] = {
                    "inputs": inputs,
                    "labels": labels
                }
    return dataset_multiple_spans
