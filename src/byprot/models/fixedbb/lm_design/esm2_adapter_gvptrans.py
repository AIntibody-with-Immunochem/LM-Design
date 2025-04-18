from dataclasses import dataclass, field
from typing import List

import torch
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import sample_from_categorical
from byprot.utils.config import compose_config as Cfg

from .modules.esm2_adapter import ESM2WithStructuralAdatper
from .modules.gvp_transformer_encoder import GVPTransformerEncoderWrapper


ESM2AdapterGVPTransConfig = Cfg(
    # encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    encoder=Cfg(
        d_model=512
    ),
    adapter_layer_indices=[-1, ],
    separate_loss=True,
    name='esm2_t33_650M_UR50D',
    dropout=0.1,
)


@register_model('esm2_adapter_gvptrans')
class ESM2AdapterGVPTrans(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ESM2AdapterGVPTransConfig

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.decoder = ESM2WithStructuralAdatper.from_pretrained(args=self.cfg, name=self.cfg.name)
        self.encoder = GVPTransformerEncoderWrapper(self.decoder.alphabet, freeze=True)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def forward(self, batch, **kwargs):
        encoder_logits, encoder_out = self.encoder(batch, return_feats=True, **kwargs)

        encoder_out['feats'] = encoder_out['feats'].detach()

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        esm_logits = self.decoder(
            tokens=init_pred,
            encoder_out=encoder_out,
        )['logits']

        if not getattr(self.cfg, 'separate_loss', False):
            logits = encoder_logits + esm_logits
            return logits, encoder_logits
        else:
            return esm_logits, encoder_logits

    def forward_encoder(self, batch):
        encoder_logits, encoder_out = self.encoder(batch, return_feats=True)

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        encoder_out['logits'] = encoder_logits
        encoder_out['init_pred'] = init_pred
        encoder_out['coord_mask'] = batch['coord_mask']
        
        # Pass the prev_token_mask to the encoder_out dictionary
        # This is the mask indicating which positions are CDR regions (masked)
        if 'prev_token_mask' in batch:
            encoder_out['prev_token_mask'] = batch['prev_token_mask']
            
        return encoder_out

    def forward_decoder(self, prev_decoder_out, encoder_out, need_attn_weights=False):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        # Define masks for all non-padding tokens and for CDR regions (masked tokens)
        all_token_mask = output_tokens.ne(self.padding_idx)  # All non-padding tokens
        cdr_mask = output_tokens.eq(self.mask_idx)  # Only CDR regions (masked tokens)

        esm_logits = self.decoder(
            # tokens=encoder_out['init_pred'],
            tokens=output_tokens,
            encoder_out=encoder_out,
        )['logits']

        if not getattr(self.cfg, 'separate_loss', False):
            logits = esm_logits + encoder_out['logits']
        else:
            logits = esm_logits  # + encoder_out['logits']

        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        # Only update tokens in CDR regions (masked positions)
        # This ensures non-CDR regions remain fixed during prediction
        if 'prev_token_mask' in encoder_out:
            # If we have explicit CDR mask information from the batch
            cdr_mask = encoder_out['prev_token_mask']
            output_tokens.masked_scatter_(cdr_mask, _tokens[cdr_mask])
            output_scores.masked_scatter_(cdr_mask, _scores[cdr_mask])
        else:
            # Fall back to using mask_idx as indicator of CDR regions
            output_tokens.masked_scatter_(cdr_mask, _tokens[cdr_mask])
            output_scores.masked_scatter_(cdr_mask, _scores[cdr_mask])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            step=step + 1,
            max_step=max_step,
            history=history
        )

    def initialize_output_tokens(self, batch, encoder_out):
        mask = encoder_out.get('coord_mask', None)

        prev_tokens = batch['prev_tokens']
        prev_token_mask = batch['prev_token_mask']
        # lengths = prev_tokens.ne(self.padding_idx).sum(1)

        # initial_output_tokens = torch.full_like(prev_tokens, self.padding_idx)
        # initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.mask_idx)
        # initial_output_tokens[:, 0] = self.cls_idx
        # initial_output_tokens.scatter_(1, lengths[:, None] - 1, self.eos_idx)

        # initial_output_tokens = encoder_out['init_pred'].clone()
        initial_output_tokens = torch.where(
            prev_token_mask, encoder_out['init_pred'], prev_tokens)
        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores
