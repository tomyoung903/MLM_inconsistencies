from typing import List, Dict, Any, Iterable, Union, Tuple # for type hinting
import torch
from tqdm import tqdm
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss



def calculate_num_spans(input_ids: torch.Tensor, span_length: int, gap_between_spans: int, auto_ratio: float = 0.3, max_num_spans = 99):
    '''Calculate the number of spans for the given example_id, span_length, and gap_between_spans. Minimum 1.'''
    considered_length = len(input_ids) - 12 # excluding extra_id_0, eos_token_id, sentinel token, and the first few (~8) tokens
    num_spans = int(auto_ratio * considered_length // (span_length + gap_between_spans))
    num_spans = min(num_spans, max_num_spans)
    num_spans = max(num_spans, 1)
    return num_spans


def create_multiple_span_sample(tokenizer,
                                input_ids: torch.Tensor, # 1D
                                labels: torch.Tensor, # 1D
                                span_length: int = 5,
                                gap_between_spans: int = 5,
                                num_spans: int = 2,
                                return_tensor: str = "inputs"):
    '''
    This function is a generalization of create_middle_off_sample. It manipulates the input sequence by replacing spans with unique <extra_id_n> tokens for each span.
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
    inputs: torch.Tensor - The input sequence.
    labels: torch.Tensor - The corresponding labels.
    span_length: int - The length of each span to be replaced.
    gap_between_spans: int - The length of the gap between spans.
    num_spans: int - The number of spans to replace.
    '''

    # input_ids = tokenizer(inputs, return_tensors="pt").input_ids[0]
    total_length = len(input_ids) - 2  # excluding extra_id_0 and eos_token_id

    # Calculate the starting point for the first spans
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


def create_multiple_span_sample_from_batch(
        tokenizer,
        input_ids: torch.Tensor, # 1D
        completions_batch: List[torch.Tensor], 
        span_length: int = 5,
        gap_between_spans: int = 5,  
        num_spans: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids_return = create_multiple_span_sample(
        tokenizer, 
        input_ids,
        completions_batch[0], # any completion from the batch is fine
        span_length, 
        gap_between_spans, 
        num_spans, 
        return_tensor="inputs"
    ).unsqueeze(0)
    labels = [
        create_multiple_span_sample(
            tokenizer,
            input_ids,
            completion,
            span_length,
            gap_between_spans,
            num_spans,
            "labels"
        )
        for completion in completions_batch
    ]
    labels_return = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids_return, labels_return


def create_offset_sample(input_ids: torch.Tensor, # 2D: 1*len
                        labels: torch.Tensor, # 1D
                        offset=0,
                        to_gpu=False) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Move the last offset tokens from input_ids to the front of labels.

    input_ids (1*len) == input_regular_tokens, extra_id_0, eos_token_id
    
    labels (len) == extra_id_0 + labels_regular_tokens

    Returns:

    (input_ids, labels) applied offset; input_ids is 2D Tensor and labels is 1D Tensor
    '''
    labels = labels.unsqueeze(0)
    assert offset != 0

    # when offset is used, we move the last offset tokens from input_ids to the front of labels.
    to_move = input_ids[0][-offset-2:-2] # the last two tokens are <extra_id_0> and <eos> and not moved
    labels = torch.cat((labels[0][0].unsqueeze(0), to_move, labels[0][1:]), dim=0) # the first token is <extra_id_0> and not moved
    input_ids = torch.cat((input_ids[0][:-offset-2], input_ids[0][-2:]), dim=0)
    input_ids = input_ids.unsqueeze(0)
        
    if to_gpu:
        return (input_ids.cuda(), labels.cuda())
    else:
        return (input_ids, labels)


def create_offset_sample_from_batch(
        tokenizer,
        input_ids: torch.Tensor, #  2D: 1*len
        completions_batch: List[torch.Tensor], 
        offset: int = 0,
        to_gpu: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    input_ids: (1*len) 
    completions_batch: List of 1D tensors
    '''

    input_ids_return = create_offset_sample(
        input_ids,
        completions_batch[0], # any completion from the batch is fine
        offset,
        to_gpu
    )[0]

    labels_return = [
        create_offset_sample(
            input_ids,
            completion,
            offset,
            to_gpu
        )[1]
        for completion in completions_batch
    ]
    labels_return = torch.nn.utils.rnn.pad_sequence(
                        labels_return, 
                        batch_first=True, 
                        padding_value=tokenizer.pad_token_id
                    )
    return input_ids_return, labels_return


def multi_labels_forward(
    model,
    input_ids: torch.Tensor, # 2D: 1 * max_len
    labels: torch.Tensor, # 2D: batch_size * max_len
    use_cache: bool = None,
    return_dict: bool = None
) -> Seq2SeqLMOutput:
    r"""
    Sometimes the input_ids are the same for multiple labels. This function is to avoid repeated encoder computation.
    Copied from T5ForConditionalGeneration.forward() from transformers/models/t5/modeling_t5.py with minor changes.

    Args:
        input_ids (`torch.Tensor` of shape `(1, sequence_length)`)

        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`)

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

    # save hidden states to a .torch file
    # torch.save(hidden_states, 'hidden_states_batch.pt')


    if model.model_parallel:
        torch.cuda.set_device(model.decoder.first_device)

    # get decoder inputs from shifting lm labels to the right (add pad_token_id at the beginning)
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
