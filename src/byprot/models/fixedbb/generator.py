import itertools
import math
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Mapping

import torch
from torch import nn


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def _cdr_skeptical_unmasking(output_scores, original_mask, p):
    """
    Similar to _skeptical_unmasking but only considers positions that were originally masked.
    This ensures that only the CDR regions are being redesigned while the rest remains fixed.
    
    Args:
        output_scores: Confidence scores for each position
        original_mask: Boolean mask indicating which positions were originally masked (CDR regions)
        p: Proportion of tokens to remask
    
    Returns:
        A boolean mask indicating which positions should be remasked
    """
    # Create a copy of output_scores
    masked_scores = output_scores.clone()
    
    # Set scores for non-CDR regions to a very high value so they won't be remasked
    masked_scores.masked_fill_(~original_mask, float('inf'))
    
    # Sort indices by scores (ascending)
    sorted_index = masked_scores.sort(-1)[1]
    
    # Calculate how many tokens to remask based on the number of originally masked tokens
    boundary_len = (
        (original_mask.sum(1, keepdim=True).type_as(output_scores)) * p
    ).long()
    
    # Create mask for tokens to be remasked (lowest confidence scores within CDR regions)
    skeptical_mask = new_arange(original_mask) < boundary_len
    
    # Scatter the mask according to sorted indices
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask) & original_mask


def exists(obj):
    return obj is not None


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def maybe_remove_batch_dim(tensor):
    if len(tensor.shape) > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


class IterativeRefinementGenerator(object):
    def __init__(self,
                 alphabet=None,
                 max_iter=1,
                 strategy='denoise',
                 temperature=None,
                 **kwargs
                 ):

        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx

        self.max_iter = max_iter
        self.strategy = strategy
        # print(f'[DEBUG - designer.py] Using Strategy: {self.strategy}')
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, model, batch, alphabet=None, 
                 max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
                 need_attn_weights=False):
        alphabet = alphabet or self.alphabet
        max_iter = max_iter or self.max_iter
        strategy = strategy or self.strategy
        temperature = temperature or self.temperature

        # 0) encoding
        encoder_out = model.forward_encoder(batch)

        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = model.initialize_output_tokens(
            batch, encoder_out=encoder_out)
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )
        
        # Debug: Print initial sequence with masks
        alphabet = alphabet or self.alphabet
        print("\n===== ITERATIVE REFINEMENT DEBUG =====")
        initial_seq = alphabet.decode(initial_output_tokens, return_as='str', remove_special=False)
        print(f"Initial sequence (iteration 0):")
        for seq in initial_seq:
            # Replace mask tokens with '-' for better visualization
            masked_seq = seq.replace(alphabet.get_tok(alphabet.mask_idx), '-')
            print(masked_seq)

        if need_attn_weights:
            attns = [] # list of {'in', 'out', 'attn'} for all iteration

        if strategy == 'discrete_diffusion':
            prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(batch['prev_tokens'])

        # iterative refinement
        for step in range(max_iter):
            
            # For cdr_mask_predict strategy, ensure non-CDR regions remain fixed before prediction
            if strategy == 'cdr_mask_predict' and replace_visible_tokens:
                # Get the original mask and tokens
                visible_token_mask = ~batch['prev_token_mask']
                visible_tokens = batch['prev_tokens']
                
                # Replace non-CDR tokens with original values before prediction
                prev_decoder_out['output_tokens'] = torch.where(
                    visible_token_mask, visible_tokens, prev_decoder_out['output_tokens'])
                
                print(f"\n----- Pre-prediction fixup for iteration {step+1}/{max_iter} -----")
                print("Ensuring non-CDR regions remain fixed with original amino acids before prediction")

            # 2.1: predict
            decoder_out = model.forward_decoder(
                prev_decoder_out=prev_decoder_out,
                encoder_out=encoder_out,
                need_attn_weights=need_attn_weights
            )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']
            
            # Debug: Print sequence after prediction (before remasking)
            print(f"\n----- Iteration {step+1}/{max_iter} -----")
            predicted_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
            
            # Also get the original sequence for comparison
            original_seq = alphabet.decode(batch['prev_tokens'], return_as='str', remove_special=False)
            
            print(f"After prediction (before remasking):")
            
            # Print both sequences for comparison
            for i, (orig, pred) in enumerate(zip(original_seq, predicted_seq)):
                # Replace mask tokens with '-' for better visualization
                orig_masked = orig.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                pred_masked = pred.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                
                # For cdr_mask_predict strategy, show differences between original and predicted
                if strategy == 'cdr_mask_predict':
                    # Create a diff string that shows where predictions differ from original
                    diff_str = ""
                    mask_tensor = batch['prev_token_mask'][i]
                    seq_len = min(len(orig_masked), len(pred_masked), mask_tensor.size(0))
                    
                    for j in range(seq_len):
                        try:
                            if mask_tensor[j]:  # If this position was masked (CDR region)
                                diff_str += pred_masked[j]  # Show the predicted amino acid
                            else:
                                # For non-CDR regions, check if prediction matches original
                                if orig_masked[j] == pred_masked[j]:
                                    diff_str += " "  # Match - show space
                                else:
                                    diff_str += "!"  # Mismatch - show !
                        except IndexError:
                            # Handle any potential index errors
                            break
                    
                    print("Original: " + orig_masked)
                    print("Predicted: " + pred_masked)
                    print("Diff     : " + diff_str)
                    print("(Spaces indicate matching non-CDR positions, '!' indicates incorrect predictions in non-CDR regions)")
                else:
                    # For other strategies, just show the predicted sequence
                    print(pred_masked)

            # 2.2: re-mask skeptical parts of low confidence
            # skeptical decoding (depend on the maximum decoding steps.)
            if (
                strategy == 'mask_predict'
                and (step + 1) < max_iter
            ):
                skeptical_mask = _skeptical_unmasking(
                    output_scores=output_scores,
                    output_masks=output_tokens.ne(self.padding_idx),  # & coord_mask,
                    p=1 - (step + 1) / max_iter
                )
                
                # Store tokens before remasking for comparison
                before_remask = output_tokens.clone()
                
                output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
                output_scores.masked_fill_(skeptical_mask, 0.0)
                
                # Debug: Print sequence after remasking
                after_remask_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                print(f"After remasking (p={1 - (step + 1) / max_iter:.2f}):")
                for seq in after_remask_seq:
                    # Replace mask tokens with '-' for better visualization
                    masked_seq = seq.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                    print(masked_seq)
            
            elif (
                strategy == 'cdr_mask_predict'
                and (step + 1) < max_iter
            ):
                # Get the original mask from the batch (which positions were originally masked)
                original_mask = batch['prev_token_mask']
                
                # Calculate proportion to remask (decreases with each iteration)
                p_value = 1 - (step + 1) / max_iter
                
                # Apply CDR-specific skeptical unmasking
                skeptical_mask = _cdr_skeptical_unmasking(
                    output_scores=output_scores,
                    original_mask=original_mask,
                    p=p_value
                )
                
                # Store tokens before remasking for comparison
                before_remask = output_tokens.clone()
                
                # Apply the mask
                output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
                output_scores.masked_fill_(skeptical_mask, 0.0)
                
                # Debug: Print sequence after remasking
                after_remask_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                before_remask_seq = alphabet.decode(before_remask, return_as='str', remove_special=False)
                
                print(f"After CDR-specific remasking (p={p_value:.2f}):")
                print(f"Number of remasked tokens: {skeptical_mask.sum().item()} out of {original_mask.sum().item()} originally masked tokens")
                
                for i, (before, after) in enumerate(zip(before_remask_seq, after_remask_seq)):
                    # Replace mask tokens with '-' for better visualization
                    before_masked = before.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                    after_masked = after.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                    
                    # Create a remasking indicator string
                    remask_indicator = ""
                    seq_len = min(len(before_masked), len(after_masked))
                    
                    try:
                        for j in range(seq_len):
                            if after_masked[j] == '-' and before_masked[j] != '-':
                                remask_indicator += "^"  # Position was remasked
                            else:
                                remask_indicator += " "  # Position was not remasked
                    except IndexError:
                        # Handle any potential index errors
                        pass
                    
                    print("Before remasking: " + before_masked)
                    print("After remasking : " + after_masked)
                    print("Remasked       : " + remask_indicator)
                    print("(^ indicates positions that were remasked based on confidence scores)")

            elif strategy == 'denoise' or strategy == 'no':
                print("Strategy: denoise or no - No remasking performed")
                pass
            elif strategy == 'discrete_diffusion':
                print("Strategy: discrete_diffusion - No remasking performed")
                pass
            else:
                print(f"Strategy: {strategy} - No remasking performed")
                pass

            if replace_visible_tokens:
                visible_token_mask = ~batch['prev_token_mask']
                visible_tokens = batch['prev_tokens']
                output_tokens = torch.where(
                    visible_token_mask, visible_tokens, output_tokens)
                
                if strategy == 'cdr_mask_predict':
                    print("\nRestoring non-CDR regions to original amino acids after prediction")
                    # Show which positions were restored
                    restored_count = visible_token_mask.sum().item()
                    total_count = visible_token_mask.numel()
                    print(f"Restored {restored_count} non-CDR positions out of {total_count} total positions")

            if need_attn_weights:
                attns.append(
                    dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
                         output=maybe_remove_batch_dim(output_tokens),
                         attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
                )

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        # skeptical_mask = _skeptical_unmasking(
        #     output_scores=output_scores,
        #     output_masks=output_tokens.ne(self.padding_idx),  # & coord_mask,
        #     p=0.08
        # )

        # output_tokens.masked_fill_(skeptical_mask, self.alphabet.unk_idx)
        # output_scores.masked_fill_(skeptical_mask, 0.0)
        decoder_out = prev_decoder_out
        
        # Debug: Print final sequence
        final_seq = alphabet.decode(decoder_out['output_tokens'], return_as='str', remove_special=False)
        print("\n----- Final Sequence -----")
        for seq in final_seq:
            # Replace mask tokens with '-' for better visualization
            masked_seq = seq.replace(alphabet.get_tok(alphabet.mask_idx), '-')
            print(masked_seq)
        print("===== END OF ITERATIVE REFINEMENT DEBUG =====\n")

        if need_attn_weights:
            return decoder_out['output_tokens'], decoder_out['output_scores'], attns
        return decoder_out['output_tokens'], decoder_out['output_scores']


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores
