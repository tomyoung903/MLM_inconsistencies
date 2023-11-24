'''Utility functions to process the LLM model output on the lambada dataset to help with evaluation'''
from typing import List, Dict, Any, Iterable # for type hinting
from torch import Tensor
import numpy as np
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json


class LambadaOutputProcessor:
    '''Process the output (completion) of the LLM model on the lambada dataset to obtain valid last words.'''
    def __init__(self, 
                 tokenizer, 
                 ul2_mode: str,
                 lambada_test_set_path: str,):
        self.tokenizer = tokenizer
        self.ENDING_PUNCTUATIONS = ',!.:;?' # If the model generates one, it is considered that the sentence is complete and we can parse for the last word
        self.vocab = tokenizer.get_vocab()
        self.ENDING_PUNCTUATIONS_IDS_LIST = [self.vocab[p] for p in self.ENDING_PUNCTUATIONS]
        with open(lambada_test_set_path, "r") as f:
            lambada = [json.loads(line) for line in f.readlines()]


        # To use the NLG mode of UL2, append [NLG] to the beginning of each input, and <extra_id_0> to the end
        lambada = [
            {
                "inputs_pretokenized": ul2_mode + " " + x['inputs_pretokenized'] + " <extra_id_0>",
                "targets_pretokenized": x['targets_pretokenized']
            } 
            for x in lambada
        ]

        self.dataset = lambada

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



    def before_first_punc(self, completion_ids: List[Tensor]):
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
        Remove a random span in the middle of the input and replace it with <extra_id_0>, 
        the old <extra_id_0> should be replaced with <extra_id_1>
        '''
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids

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

        input_regular_tokens = input_ids[0][:-2]
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


    def get_middle_off_samples(self,
                                id_to_completions: Dict[int, List[torch.Tensor]],
                                range_middle_span_length: Iterable[int],
                                range_middle_to_end_gap: Iterable[int],
                                to_gpu=False):
        '''Apply create_middle_off_sample to all the completions of each example by calling create_middle_off_sample'''
        dataset_middle_off = {}
        for example_id in tqdm(range(len(id_to_completions))):
            # skip if there is no completion
            if len(id_to_completions[example_id]) == 0:
                continue
            for middle_span_length in range_middle_span_length:
                for middle_to_end_gap in range_middle_to_end_gap:
                    inputs = self.create_middle_off_sample(
                        self.dataset[example_id]['inputs_pretokenized'],
                        id_to_completions[example_id][0], # any completion is fine to get the input_ids
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
                        for completion in id_to_completions[example_id]
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
                    
    def is_correct_completion(self, example_index:int, completion:torch.Tensor):
        if not isinstance(completion, torch.Tensor):
            return False
        completion_string = self.tokenizer.decode(completion)
        if not isinstance(completion_string, str):
            return False
        word = self.get_word_from_completion(completion_string)
        if not isinstance(word, str):
            return False
        if word == self.dataset[example_index]['targets_pretokenized'][0]:
            return True
        

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
