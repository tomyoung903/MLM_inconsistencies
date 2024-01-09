'''Utility functions to process the LLM model output on the lambada dataset to help with evaluation'''
from typing import List, Dict, Any, Iterable, Union, Tuple # for type hinting
from torch import Tensor
import numpy as np
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json


class LambadaProcessor:
    '''Process the output (completion) of the LLM model on the lambada dataset to obtain valid last words.'''
    def __init__(self, 
                 tokenizer, 
                 ul2_mode: str,
                 lambada_dataset_path: str,
                 rm_punc_space:bool):
        '''
        Args:
            tokenizer: the tokenizer used to tokenize the input
            ul2_mode: "[NLG]", "[NLU]" or "[S2S]"
            lambada_dataset_path: the path to the lambada dataset
            rm_punc_space: whether to remove the spaces before punctuations in the completion
        '''
        
        self.tokenizer = tokenizer
        self.ENDING_PUNCTUATIONS = ',!.:;?' # If the model generates one, it is considered that the sentence is complete and we can parse for the last word
        self.vocab = tokenizer.get_vocab()
        self.ENDING_PUNCTUATIONS_IDS_LIST = [self.vocab[p] for p in self.ENDING_PUNCTUATIONS]
        with open(lambada_dataset_path, "r") as f:
            lambada = [json.loads(line) for line in f.readlines()]


        # remove spaces before punctuations
        if rm_punc_space:
            lambada = [
                {
                    "inputs_pretokenized": self._remove_spaces_before_puncs(x['inputs_pretokenized']),
                    "targets_pretokenized": x['targets_pretokenized']
                } 
                for x in lambada
            ]

        # append ul2_mode to the beginning of each input, and <extra_id_0> to the end
        lambada = [
            {
                "inputs_pretokenized": ul2_mode + " " + x['inputs_pretokenized'] + " <extra_id_0>",
                "targets_pretokenized": x['targets_pretokenized']
            } 
            for x in lambada
        ]

        
        self.dataset = lambada

    def _remove_spaces_before_puncs(self, paragraph: str):
        '''Remove the spaces before punctuations in the paragraph'''
        PUNCS = [',', '!', '.', ':', ';', '?', 'n\'t', '\'d', '\'s', '\'m', '\'ll']
        # check if there is a space before the punc, if so remove the space
        # Example 'i love you !' -> 'i love you!'
        for punc in PUNCS:
            paragraph = paragraph.replace(" " + punc, punc)
        return paragraph

    def get_word_from_completion(self, completion: str):
        '''Get the last word from the given completion, if there is a valid one. Return the word.'''
        found = False
        # if a punctuation can be found in the completion, get the string before the punctuation
        for i in range(len(completion)):
            if completion[i] in self.ENDING_PUNCTUATIONS:
                word = completion[:i]
                found = True
                break
        if not found:
            return None

        '''postprocess the string to remove invalidities'''
        # if the word starts with <pad>, remove it
        word = word[5:] if word.startswith("<pad>") else word
        # if the word starts with <extra_id_0>, remove it
        word = word[12:] if word.startswith("<extra_id_0>") else word
        # if the word starts with ' ', remove it
        word = word[1:] if word.startswith(" ") else word
        # if the word ends with ' ', remove it
        word = word[:-1] if word.endswith(" ") else word
        # if the word is empty, discount it
        word = word if word != "" else None
        # if there are multiple words in it, discount it
        if word:
            word = word if len(word.split(" ")) == 1 else None
        return word


    def get_words_from_completions(self, completions: List[str]):
        '''Get the last word from each of the given completions,  Return all the words.'''
        # if a punctuation can be found in the completion, get the word before the punctuation
        words = []
        for completion in completions:
            word = self.get_word_from_completion(completion)
            if word:
                words.append(word)
        return words



    def get_punctuated_words(self, completions: List[str]):
        '''given a list of completions (completions by the LLM), return a list of word-punc pairs'''

        # if a punctuation can be found in the completion, get the word and the punc
        words = []
        for completion in completions:
            # find the punctuation
            for i in range(len(completion)):
                if completion[i] in self.ENDING_PUNCTUATIONS:
                    word = completion[:i+1]
                    words.append(word)
                    # show(words)
                    break
        
        # if the word starts with <pad>, remove the <pad>
        words = [word[5:] if word.startswith("<pad>") else word for word in words]
        # if the word starts with <extra_id_0>, remove the <extra_id_0>
        words = [word[12:] if word.startswith("<extra_id_0>") else word for word in words]
        # if the word starts with ' ', remove it
        words = [word[1:] if word.startswith(" ") else word for word in words]
        # if the word ends with ' ', remove it
        words = [word[:-1] if word.endswith(" ") else word for word in words]
        # if the word is empty, remove it
        words = [word for word in words if word != ""]
        # if there are multiple words in word, remove it
        words = [word for word in words if len(word.split(" ")) == 1]
        # if the length is 1, remove it (to prevent the case where it is just a punctuation)
        words = [word for word in words if len(word) > 1]
        # if the word contains <unk>, remove it
        words = [word for word in words if "<unk>" not in word]
        return list(set(words))


    def remove_pad(self, completions: List[str]):
        '''given a list of completions (completions by the LLM), remove the <pad>'''
        # if the word starts with <pad>, remove the <pad>
        completions = [completion[5:] if completion.startswith("<pad>") else completion for completion in completions]
        return completions


    def remove_pad_id(self, completions: List[Tensor]):
        '''given a list of completions of ids (completions by the LLM), remove the <pad>'''
        pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        # if the word starts with <pad>, remove the <pad>
        completions_return = []
        for completion in completions:
            if completion[0] == pad_id:
                completions_return.append(completion[1:])
            else:
                completions_return.append(completion)
        return completions_return



    def before_first_punc_including(self, completion_ids: List[Tensor]):
        '''given a list of completion_ids (completions by the LLM), return the string before (including) the first punctuation'''
        completions_ids_return = []
        for completion in completion_ids:
            for i in range(len(completion)):
                if completion[i] in self.ENDING_PUNCTUATIONS_IDS_LIST:
                    completions_ids_return.append(completion[:i+1])
                    break
        return completions_ids_return
    

    def create_offset_sample(self,
                            inputs: str, 
                            labels: torch.Tensor, 
                            offset=0,
                            to_gpu=False):
        '''
        Move the last offset tokens from input_ids to the front of labels.
        '''

        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids
        labels = labels.unsqueeze(0)
        '''
        input_ids (1*len) == input_regular_tokens, extra_id_0, eos_token_id
        labels (1*len) == extra_id_0 + labels_regular_tokens
        '''
        if offset != 0:
            # when offset is used, we move the last offset tokens from input_ids to the front of labels.
            to_move = input_ids[0][-offset-2:-2] # the last two tokens are <extra_id_0> and <eos> and not moved
            labels = torch.cat((labels[0][0].unsqueeze(0), to_move, labels[0][1:]), dim=0) # the first token is <extra_id_0> and not moved
            input_ids = torch.cat((input_ids[0][:-offset-2], input_ids[0][-2:]), dim=0)
        else:
            # squeeze the batch dimension
            labels = labels[0]
            input_ids = input_ids[0]
        if to_gpu:
            return (input_ids.cuda(), labels.cuda())
        else:
            return (input_ids, labels)

    def get_offset_samples(self,
                           id_to_completions_ids: Dict[int, List],
                           max_offset=5,
                           to_gpu=False):
        '''add offset to the samples'''
        dataset_offset = {}
        for example_id in tqdm(range(len(id_to_completions_ids))):
            if len(id_to_completions_ids[example_id]) == 0:
                continue
            for offset in range(max_offset):
                # get input_ids, which is identical for all completions
                input_ids = self.create_offset_sample(
                    self.dataset[example_id]['inputs_pretokenized'],
                    id_to_completions_ids[example_id][0], # any completion is fine to get the input_ids
                    offset=offset,
                    to_gpu=to_gpu,
                )[0]
                labels = [
                    self.create_offset_sample(
                        self.dataset[example_id]['inputs_pretokenized'],
                        completion,
                        offset=offset,
                        to_gpu=to_gpu,
                    )[1]
                    for completion in id_to_completions_ids[example_id]
                ]
                # pad into a single tensor
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                dataset_offset[(example_id, offset)] = {
                    "inputs": input_ids,
                    "labels": labels
                }
                    
        return dataset_offset
        

    def create_middle_off_sample(self,
                                inputs: str,
                                labels: torch.Tensor, # 1D
                                middle_span_length: int = 5,
                                middle_to_end_gap: int = 5,
                                return_tensor: str = "inputs"):
        '''
        Remove a span in the middle of the input and replace it with <extra_id_0>, 
        the old <extra_id_0> should be replaced with <extra_id_1>
        '''
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids # 2D: 1 * len

        '''
        input_ids (1*len) == [input_regular_tokens, extra_id_0, eos_token_id]
         (break down input_regular_tokens) -> 
          input_regular_tokens_keep_0, input_regular_tokens_move_0, input_regular_tokens_keep_1, extra_id_0, eos_token_id
         (replace input_regular_tokens_move_0 with <extra_id_0>) -> 
          input_regular_tokens_keep_0, extra_id_0, input_regular_tokens_keep_1, extra_id_1, eos_token_id
          
          
        middle_span_length: len(input_regular_tokens_move_0)
        middle_to_end_gap: len(input_regular_tokens_keep_1)
            
        labels (1D) == [extra_id_0, labels_regular_tokens]
            (add input_regular_tokens_move_0 to the front and change <extra_id_0> to <extra_id_1>) ->
            extra_id_0, input_regular_tokens_move_0, extra_id_1, labels_regular_tokens

        return_tensor: "inputs" or "labels"
        '''

        input_regular_tokens = input_ids[0][:-2] # the last two tokens are <extra_id_0>
        input_regular_tokens_keep_1 = input_regular_tokens[-middle_to_end_gap:]
        input_regular_tokens_move_0 = input_regular_tokens[-middle_to_end_gap-middle_span_length:-middle_to_end_gap]
        input_regular_tokens_keep_0 = input_regular_tokens[:-middle_to_end_gap-middle_span_length]
        
        extra_id_0 = torch.tensor([self.tokenizer.convert_tokens_to_ids("<extra_id_0>")])
        extra_id_1 = torch.tensor([self.tokenizer.convert_tokens_to_ids("<extra_id_1>")])

        if return_tensor == "inputs":
            input_ids_return = torch.cat(
                (
                    input_regular_tokens_keep_0,
                    extra_id_0,
                    input_regular_tokens_keep_1,
                    extra_id_1,
                    torch.tensor([self.tokenizer.eos_token_id])
                )
            )
            return input_ids_return
        elif return_tensor == "labels":
            labels_return = torch.cat(
                (
                    extra_id_0,
                    input_regular_tokens_move_0,
                    extra_id_1,
                    labels[1:]
                )
            )
            return labels_return
        else:
            raise ValueError("return_tensor should be either 'inputs' or 'labels'")


    def create_nlg_style_samples(self,
                                inputs: str,
                                labels: torch.Tensor, # 1D
                                span_length: int = 5,
                                gap_between_spans: int = 5,
                                num_spans: int = 2):
        '''
        Remove spans in the middle of the input and replace them with <extra_id_0>, <extra_id_1>, <extra_id_2>, ...
        # 32099: <extra_id_0>
        # 32098: <extra_id_1>
        # ...
        # 32000: <extra_id_99>
        Start from the end and work backwards.
        '''
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids



        return
        

    def create_multiple_span_sample(self,
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
            
        
        The function dynamically segments the input_regular_tokens into multiple spans and gaps, replacing each span with a corresponding <extra_id_n>.
        The labels are also adjusted to match the new structure of the input sequence.

        Parameters:
        inputs: str - The input text.
        labels: torch.Tensor - The corresponding labels.
        span_length: int - The length of each span to be replaced.
        gap_between_spans: int - The length of the gap between spans.
        num_spans: int - The number of spans to replace.
        '''

        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        total_length = len(input_ids) - 2  # excluding extra_id_0 and eos_token_id

        # Calculate the starting point for the first span
        total_span_length = (span_length + gap_between_spans) * num_spans # Total length of all spans and gaps
        start_of_span = total_length - total_span_length

        input_ids_return = input_ids[:start_of_span]
        # initialize labels_return as 1*0 tensor
        labels_return = torch.tensor([],dtype=torch.int64)
        for i in range(num_spans):
            extra_id_token = torch.tensor([self.tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")])

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
        extra_id_token = torch.tensor([self.tokenizer.convert_tokens_to_ids(f"<extra_id_{num_spans}>")])
        input_ids_return = torch.cat((input_ids_return, extra_id_token, input_ids[-1:]))

        # Add the last extra_id_token and the labels_regular_tokens to labels_return
        labels_return = torch.cat((labels_return, extra_id_token, labels[1:]))

        if return_tensor == "inputs":
            return input_ids_return
        elif return_tensor == "labels":
            return labels_return


    def get_middle_off_samples(self,
                                id_to_completions_ids: Dict[int, List[torch.Tensor]],
                                length_gap_tuples: List[tuple],
                                to_gpu=False):
        '''Apply create_middle_off_sample to all the completions of each example for every length_gap tuple'''
        dataset_middle_off = {}
        for example_id in tqdm(range(len(id_to_completions_ids))):
            # skip if there is no completion
            if len(id_to_completions_ids[example_id]) == 0:
                continue
            for (middle_span_length, middle_to_end_gap) in length_gap_tuples:
                inputs = self.create_middle_off_sample(
                    self.dataset[example_id]['inputs_pretokenized'],
                    id_to_completions_ids[example_id][0], # any completion is fine to get the input_ids
                    middle_span_length=middle_span_length,
                    middle_to_end_gap=middle_to_end_gap,
                    return_tensor="inputs"
                )
                # get the list of labels first then pad them to the same length for a large tensor
                labels = [
                    self.create_middle_off_sample(
                        self.dataset[example_id]['inputs_pretokenized'],
                        completion,
                        middle_span_length=middle_span_length,
                        middle_to_end_gap=middle_to_end_gap,
                        return_tensor="labels"
                    )
                    for completion in id_to_completions_ids[example_id]
                ]
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

                if to_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                dataset_middle_off[(example_id, middle_span_length, middle_to_end_gap)] = {
                        "inputs": inputs,
                        "labels": labels
                    }
        return dataset_middle_off


    def get_multiple_span_samples(self,
                                  id_to_completions_ids: Dict[int, List[torch.Tensor]],
                                  length_gap_num_tuple: tuple = (3, 5, None),
                                  max_num_spans = 99, # <extra_id_k> goes up to <extra_id_99>
                                  auto_ratio = 0.3, # if num_spans is None, then the no. spans decided by input length is scaled by this ratio
                                  to_gpu=False):
        '''Apply create_multiple_span_sample to all the completions of each example with length_gap_num_tuple.
        If num_spans is None, then the number of spans is decided by the length of the input.'''
        dataset_multiple_span = {}
        span_length, gap_between_spans, num_spans = length_gap_num_tuple
        is_num_spans_given = num_spans != None

        for example_id in tqdm(range(len(id_to_completions_ids))):
            # skip if there is no completion
            if len(id_to_completions_ids[example_id]) == 0:
                continue
            
            if not is_num_spans_given: # decide via length of input for each example
                input_ids = self.tokenizer(self.dataset[example_id]['inputs_pretokenized'], return_tensors="pt").input_ids[0]
                operation_length = len(input_ids) - 12  # excluding extra_id_0, eos_token_id, sentinel token, and the first few (~8) tokens 
                num_spans = int(auto_ratio * operation_length // (span_length + gap_between_spans))
                num_spans = min(num_spans, max_num_spans)

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

            # if is_num_spans_given:
            dataset_multiple_span[(example_id, span_length, gap_between_spans, num_spans)] = {
                    "inputs": inputs,
                    "labels": labels
                }
            # else:
            #     dataset_multiple_span[(example_id, span_length, gap_between_spans, 'auto')] = {
            #             "inputs": inputs,
            #             "labels": labels
            #         }
        return dataset_multiple_span


    def is_correct_result(self, example_index:int, completion_or_word:Union[torch.Tensor, str]):
        '''
        Check if the given completion_or_word is correct for the given example. If completion_or_word is a torch.Tensor, it is treated as a completion tensor, otherwise it is treated as a word string.
        '''
        if isinstance(completion_or_word, torch.Tensor):
            # treat it as a completion tensor
            completion_string = self.tokenizer.decode(completion_or_word)
            if not isinstance(completion_string, str):
                return False
            word = self.get_word_from_completion(completion_string)
            if not isinstance(word, str):
                return False
            return word == self.dataset[example_index]['targets_pretokenized'][0]
        elif isinstance(completion_or_word, str):  
            # treat it as a word string
            return completion_or_word == self.dataset[example_index]['targets_pretokenized'][0]
        else:
            raise ValueError("completion_or_word should be either a torch.Tensor or a str")


    def collate_puncs(self, avg_log_p_and_completion: List[Tuple[float, torch.Tensor]]):
        ''' This function did not help with the performance. It is not used at the moment.
        Collate different puncs for the same word: if a word has multiple puncs, then the avg_log_p's for (word, punc_k)'s are max-pooled or added up or avg'ed.Right now they are avg'ed.'''    
        word_to_avg_log_p = {}
        # print(avg_log_p_and_completion)
        for avg_log_p, completion in avg_log_p_and_completion:
            # print(avg_log_p, completion)
            word = self.get_word_from_completion(self.tokenizer.decode(completion))
            word_to_avg_log_p[word] = word_to_avg_log_p.get(word, []) + [avg_log_p]

        # convert back to list of tuples
        # print(word_to_avg_log_p)
        avg_log_p_and_word = [(sum(word_to_avg_log_p[word]) / len(word_to_avg_log_p[word]), word) for word in word_to_avg_log_p]
        return avg_log_p_and_word
    

def multi_labels_forward(
    model,
    input_ids: torch.Tensor, 
    labels: torch.Tensor,
    use_cache: bool = None,
    return_dict: bool = None
) -> Seq2SeqLMOutput:
    r"""
    Sometimes the input_ids are the same for multiple labels. This function is to avoid the repeated encoder computation.
    Copied from T5ForConditionalGeneration.forward() from transformers/models/t5/modeling_t5.py with minor changes.

    Args:
        input_ids (`torch.Tensor` of shape `(1, sequence_length)`):
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`):

    """
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict


    # Encode if needed (training, first prediction pass)
    encoder_outputs = model.encoder(
        input_ids=input_ids,
        return_dict=return_dict,
    )

    hidden_states = encoder_outputs[0]
    
    # get batch size from labels
    batch_size = labels.shape[0]

    # repeat along the batch dimension to match the number of labels 
    hidden_states = hidden_states.repeat(batch_size, 1, 1)

    if model.model_parallel:
        torch.cuda.set_device(model.decoder.first_device)

    # get decoder inputs from shifting lm labels to the right
    decoder_input_ids = model._shift_right(labels)

    # Set device for model parallelism
    if model.model_parallel:
        torch.cuda.set_device(model.decoder.first_device)
        hidden_states = hidden_states.to(model.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(model.decoder.first_device)
    # Decode
    # hidden_states.shape: batch_size * max_len * hidden_states_dim
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=hidden_states,
        use_cache=use_cache,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if model.model_parallel:
        torch.cuda.set_device(model.encoder.first_device)
        model.lm_head = model.lm_head.to(model.encoder.first_device)
        sequence_output = sequence_output.to(model.lm_head.weight.device)

    if model.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (model.model_dim**-0.5)

    lm_logits = model.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )
