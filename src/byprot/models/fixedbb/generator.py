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


def _detect_polyglycine_motifs(tokens, alphabet, min_length=2):
    """
    Detect polyglycine motifs (GG, GGG, etc.) in the sequence.
    
    Args:
        tokens: Tensor of token indices
        alphabet: The alphabet object used for encoding/decoding
        min_length: Minimum length of polyglycine motif to detect (default: 2)
        
    Returns:
        A boolean mask where True indicates positions that are part of polyglycine motifs
    """
    # Get the index of glycine in the alphabet
    glycine_idx = alphabet.get_idx('G')
    
    # Initialize mask with all False
    polyglycine_mask = torch.zeros_like(tokens, dtype=torch.bool)
    
    # For each sequence in the batch
    for i in range(tokens.size(0)):
        # Convert to list for easier processing
        seq = tokens[i].cpu().tolist()
        
        # Create a valid token mask (exclude padding and special tokens)
        padding_idx = alphabet.padding_idx
        cls_idx = getattr(alphabet, 'cls_idx', -1)
        eos_idx = getattr(alphabet, 'eos_idx', -1)
        mask_idx = getattr(alphabet, 'mask_idx', -1)
        
        # Find runs of glycine
        run_start = -1
        for j in range(len(seq)):
            # Skip padding and special tokens
            if seq[j] in [padding_idx, cls_idx, eos_idx, mask_idx]:
                # If we were in a glycine run, check if it's long enough to mark
                if run_start != -1:
                    run_length = j - run_start
                    if run_length >= min_length:
                        # Mark all positions in this run
                        polyglycine_mask[i, run_start:j] = True
                    run_start = -1
                continue
                
            if seq[j] == glycine_idx:
                if run_start == -1:
                    run_start = j
            else:
                if run_start != -1:
                    run_length = j - run_start
                    if run_length >= min_length:
                        # Mark all positions in this run
                        polyglycine_mask[i, run_start:j] = True
                    run_start = -1
        
        # Check for run at the end of sequence
        if run_start != -1:
            run_length = len(seq) - run_start
            if run_length >= min_length:
                polyglycine_mask[i, run_start:len(seq)] = True
    
    return polyglycine_mask

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
                 remask_polyglycine=True,
                 polyglycine_min_length=2,
                 max_extra_iter=5,
                 **kwargs
                 ):

        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx

        self.max_iter = max_iter
        self.strategy = strategy
        # print(f'[DEBUG - designer.py] Using Strategy: {self.strategy}')
        self.temperature = temperature
        self.remask_polyglycine = remask_polyglycine
        self.polyglycine_min_length = polyglycine_min_length
        self.max_extra_iter = max_extra_iter

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
                    
                    # Account for special tokens like <cls> that might be in the string but not in the tensor
                    # Find the offset between the string representation and the tensor
                    offset = 0
                    if orig_masked.startswith("<cls>"):
                        offset = 4  # Length of "<cls>" - 1
                    
                    seq_len = min(len(orig_masked) - offset, len(pred_masked) - offset, mask_tensor.size(0))
                    
                    # Add spaces for any special tokens at the beginning
                    diff_str += " " * offset
                    
                    for j in range(seq_len):
                        try:
                            tensor_idx = j  # Index in the tensor
                            string_idx = j + offset  # Index in the string
                            
                            if tensor_idx < mask_tensor.size(0) and mask_tensor[tensor_idx]:  # If this position was masked (CDR region)
                                diff_str += pred_masked[string_idx]  # Show the predicted amino acid
                            else:
                                # For non-CDR regions, check if prediction matches original
                                if string_idx < len(orig_masked) and string_idx < len(pred_masked):
                                    if orig_masked[string_idx] == pred_masked[string_idx]:
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
                
                # Check for polyglycine motifs if enabled
                polyglycine_mask = torch.zeros_like(skeptical_mask)
                if getattr(self, 'remask_polyglycine', True):
                    # Only consider polyglycine motifs within CDR regions
                    polyglycine_mask = _detect_polyglycine_motifs(
                        output_tokens,
                        alphabet,
                        min_length=getattr(self, 'polyglycine_min_length', 2)
                    ) & original_mask
                    
                    if polyglycine_mask.any():
                        print(f"\nDetected polyglycine motifs in CDR regions!")
                        # Count polyglycine motifs
                        polyglycine_count = polyglycine_mask.sum().item()
                        print(f"Found {polyglycine_count} positions in polyglycine motifs")
                
                # Combine masks: remask both low-confidence positions and polyglycine motifs
                combined_mask = skeptical_mask | polyglycine_mask
                
                # Apply the combined mask
                output_tokens.masked_fill_(combined_mask, self.mask_idx)
                output_scores.masked_fill_(combined_mask, 0.0)
                
                # Debug: Print sequence after remasking
                after_remask_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                before_remask_seq = alphabet.decode(before_remask, return_as='str', remove_special=False)
                
                # Calculate statistics for reporting
                confidence_remasked = skeptical_mask.sum().item()
                polyglycine_remasked = polyglycine_mask.sum().item()
                total_remasked = combined_mask.sum().item()
                
                print(f"After CDR-specific remasking (p={p_value:.2f}):")
                print(f"Number of remasked tokens: {total_remasked} out of {original_mask.sum().item()} originally masked tokens")
                print(f"  - {confidence_remasked} tokens remasked based on confidence scores")
                if polyglycine_remasked > 0:
                    print(f"  - {polyglycine_remasked} additional tokens remasked from polyglycine motifs")
                
                for i, (before, after) in enumerate(zip(before_remask_seq, after_remask_seq)):
                    # Replace mask tokens with '-' for better visualization
                    before_masked = before.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                    after_masked = after.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                    
                    # Create a remasking indicator string
                    remask_indicator = ""
                    polyg_indicator = ""
                    seq_len = min(len(before_masked), len(after_masked))
                    
                    try:
                        for j in range(seq_len):
                            if j < skeptical_mask.size(1) and j < polyglycine_mask.size(1):
                                if skeptical_mask[i, j]:
                                    remask_indicator += "^"  # Position was remasked based on confidence
                                else:
                                    remask_indicator += " "
                                    
                                if polyglycine_mask[i, j]:
                                    polyg_indicator += "G"  # Position was remasked due to polyglycine
                                else:
                                    polyg_indicator += " "
                            else:
                                remask_indicator += " "
                                polyg_indicator += " "
                    except IndexError:
                        # Handle any potential index errors
                        pass
                    
                    print("Before remasking: " + before_masked)
                    print("After remasking : " + after_masked)
                    print("Confidence mask: " + remask_indicator)
                    if polyglycine_remasked > 0:
                        print("Polyglycine mask: " + polyg_indicator)
                    print("(^ indicates positions remasked based on confidence scores)")
                    if polyglycine_remasked > 0:
                        print("(G indicates positions remasked due to being part of polyglycine motifs)")

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

        # Check if we need to extend iterations due to remaining polyglycine motifs
        if (strategy == 'cdr_mask_predict' and
            getattr(self, 'remask_polyglycine', True) and
            getattr(self, 'max_extra_iter', 5) > 0):
            
            # Check for polyglycine motifs in the final sequence
            original_mask = batch['prev_token_mask']
            polyglycine_mask = _detect_polyglycine_motifs(
                output_tokens,
                alphabet,
                min_length=getattr(self, 'polyglycine_min_length', 2)
            ) & original_mask
            
            if polyglycine_mask.any():
                polyglycine_count = polyglycine_mask.sum().item()
                print(f"\n----- Extending iterations to remove remaining polyglycine motifs -----")
                print(f"Found {polyglycine_count} positions in polyglycine motifs after {max_iter} iterations")
                print(f"Will attempt up to {self.max_extra_iter} additional iterations")
                
                # Decode sequences to visualize polyglycine motifs
                current_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                
                # Identify and display the polyglycine motifs
                print("\nPolyglycine motifs detected in CDR regions:")
                for i, seq in enumerate(current_seq):
                    # Replace mask tokens with '-' for better visualization
                    seq_display = seq.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                    
                    # Create a marker string to highlight polyglycine motifs
                    pg_marker = ""
                    for j in range(len(seq_display)):
                        if j < polyglycine_mask.size(1) and polyglycine_mask[i, j]:
                            pg_marker += "^"
                        else:
                            pg_marker += " "
                    
                    print(f"Sequence: {seq_display}")
                    print(f"Polyglycine: {pg_marker}")
                    print(f"(^ indicates positions that are part of polyglycine motifs)")
                
                # Store tokens before extension for comparison
                before_extension = output_tokens.clone()
                
                # Perform additional iterations
                for extra_step in range(self.max_extra_iter):
                    print(f"\n===== EXTENSION ITERATION {extra_step+1}/{self.max_extra_iter} =====")
                    
                    # Show which positions are being remasked
                    before_remask_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                    
                    # Mask the polyglycine motifs
                    output_tokens.masked_fill_(polyglycine_mask, self.mask_idx)
                    output_scores.masked_fill_(polyglycine_mask, 0.0)
                    
                    after_remask_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                    
                    print("Remasking polyglycine motifs for this extension iteration:")
                    for i, (before, after) in enumerate(zip(before_remask_seq, after_remask_seq)):
                        # Replace mask tokens with '-' for better visualization
                        before_masked = before.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                        after_masked = after.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                        
                        # Create a remasking indicator string
                        remask_indicator = ""
                        for j in range(min(len(before_masked), len(after_masked))):
                            if after_masked[j] == '-' and before_masked[j] != '-':
                                remask_indicator += "^"  # Position was remasked
                            else:
                                remask_indicator += " "  # Position was not remasked
                        
                        print(f"Before remasking: {before_masked}")
                        print(f"After remasking : {after_masked}")
                        print(f"Remasked       : {remask_indicator}")
                        print(f"(^ indicates positions that were remasked for this extension iteration)")
                    
                    # Update decoder out for next iteration
                    prev_decoder_out.update(
                        output_tokens=output_tokens,
                        output_scores=output_scores,
                        step=max_iter + extra_step,
                        max_step=max_iter + self.max_extra_iter
                    )
                    
                    # Ensure non-CDR regions remain fixed before prediction
                    if replace_visible_tokens:
                        visible_token_mask = ~batch['prev_token_mask']
                        visible_tokens = batch['prev_tokens']
                        prev_decoder_out['output_tokens'] = torch.where(
                            visible_token_mask, visible_tokens, prev_decoder_out['output_tokens'])
                        
                        print(f"\n----- Pre-prediction fixup for extension iteration {extra_step+1}/{self.max_extra_iter} -----")
                        print("Ensuring non-CDR regions remain fixed with original amino acids before prediction")
                    
                    # Predict
                    decoder_out = model.forward_decoder(
                        prev_decoder_out=prev_decoder_out,
                        encoder_out=encoder_out,
                        need_attn_weights=need_attn_weights
                    )
                    
                    output_tokens = decoder_out['output_tokens']
                    output_scores = decoder_out['output_scores']
                    
                    # Show prediction results
                    predicted_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                    print(f"\nPrediction results for extension iteration {extra_step+1}:")
                    for i, seq in enumerate(predicted_seq):
                        # Replace mask tokens with '-' for better visualization
                        seq_display = seq.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                        print(f"Predicted: {seq_display}")
                    
                    # Restore non-CDR regions
                    if replace_visible_tokens:
                        visible_token_mask = ~batch['prev_token_mask']
                        visible_tokens = batch['prev_tokens']
                        output_tokens = torch.where(
                            visible_token_mask, visible_tokens, output_tokens)
                    
                    # Check if polyglycine motifs are gone
                    prev_polyglycine_mask = polyglycine_mask.clone()
                    polyglycine_mask = _detect_polyglycine_motifs(
                        output_tokens,
                        alphabet,
                        min_length=getattr(self, 'polyglycine_min_length', 2)
                    ) & original_mask
                    
                    polyglycine_count = polyglycine_mask.sum().item()
                    prev_count = prev_polyglycine_mask.sum().item()
                    
                    print(f"\n----- Extension iteration {extra_step+1}/{self.max_extra_iter} results -----")
                    print(f"Previous polyglycine positions: {prev_count}")
                    print(f"Remaining polyglycine positions: {polyglycine_count}")
                    
                    if polyglycine_count < prev_count:
                        print(f"Progress: Removed {prev_count - polyglycine_count} polyglycine positions in this iteration")
                    elif polyglycine_count > prev_count:
                        print(f"Warning: Generated {polyglycine_count - prev_count} new polyglycine positions in this iteration")
                    else:
                        print(f"No change in polyglycine positions in this iteration")
                    
                    # If there are still polyglycine motifs, show their locations
                    if polyglycine_count > 0:
                        after_iter_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                        print("\nRemaining polyglycine motifs:")
                        for i, seq in enumerate(after_iter_seq):
                            # Replace mask tokens with '-' for better visualization
                            seq_display = seq.replace(alphabet.get_tok(alphabet.mask_idx), '-')
                            
                            # Create a marker string to highlight polyglycine motifs
                            pg_marker = ""
                            for j in range(len(seq_display)):
                                if j < polyglycine_mask.size(1) and polyglycine_mask[i, j]:
                                    pg_marker += "^"
                                else:
                                    pg_marker += " "
                            
                            print(f"Sequence: {seq_display}")
                            print(f"Polyglycine: {pg_marker}")
                    
                    # Update for next iteration
                    prev_decoder_out.update(
                        output_tokens=output_tokens,
                        output_scores=output_scores,
                        step=max_iter + extra_step + 1,
                        history=decoder_out['history']
                    )
                    
                    # Break if no more polyglycine motifs
                    if polyglycine_count == 0:
                        print(f"Successfully removed all polyglycine motifs after {extra_step+1} additional iterations")
                        break
                
                # If we still have polyglycine motifs after all extra iterations
                if polyglycine_mask.any():
                    print(f"\nWARNING: Could not remove all polyglycine motifs after {self.max_extra_iter} additional iterations")
                    print(f"Remaining polyglycine positions: {polyglycine_mask.sum().item()}")
                
                # Compare before and after extension
                after_extension_seq = alphabet.decode(output_tokens, return_as='str', remove_special=False)
                before_extension_seq = alphabet.decode(before_extension, return_as='str', remove_special=False)
                
                print("\n----- Sequence changes after extension iterations -----")
                for i, (before, after) in enumerate(zip(before_extension_seq, after_extension_seq)):
                    print(f"Before extension: {before}")
                    print(f"After extension : {after}")
                    
                    # Create a diff string
                    diff_str = ""
                    for j in range(min(len(before), len(after))):
                        if before[j] != after[j]:
                            diff_str += "^"
                        else:
                            diff_str += " "
                    print(f"Changes        : {diff_str}")
                    print("(^ indicates positions that changed during extension iterations)")
        
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
