from omegaconf import OmegaConf, DictConfig

from byprot import utils
from byprot.datamodules.datasets import Alphabet, DataProcessor
from byprot.utils import io
from byprot.utils.config import compose_config as Cfg
from byprot.models.fixedbb.generator import IterativeRefinementGenerator
from pathlib import Path
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from collections import namedtuple

GenOut = namedtuple(
    'GenOut', 
    ['output_tokens', 'output_scores', 'attentions']
)



def _full_mask(target_tokens, coord_mask, alphabet):
    target_mask = (
        target_tokens.ne(alphabet.padding_idx)  # & mask
        & target_tokens.ne(alphabet.cls_idx)
        & target_tokens.ne(alphabet.eos_idx)
    )
    _tokens = target_tokens.masked_fill(
        target_mask, alphabet.mask_idx
    )
    _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
    return _tokens, _mask


class Designer:
    _default_cfg = Cfg(
        cuda=False,
        generator=Cfg(
            max_iter=1,
            strategy='denoise',  # ['denoise' | 'mask_predict']
            # replace_visible_tokens=False,
            temperature=0,
            eval_sc=False,  
        )
    )

    def __init__(
        self,
        experiment_path,
        cfg: DictConfig = None
    ):
        self.experiment_path = experiment_path
        self.cfg = cfg

        self._initialize()

    def _initialize(self):
        # print('[DEBUG - designer.py] Initializing class Designer by loading experiment...')
        pl_task, exp_cfg = utils.load_from_experiment(
            self.experiment_path)
        self.exp_cfg = exp_cfg

        self.model = pl_task.model
        self.model.eval()

        if self.cfg.cuda: 
            self._cuda()

        # print(f'[DEBUG - designer.py] Setting self.alphabet to {pl_task.alphabet}')
        # print(f'[DEBUG - designer.py] pl_task is {pl_task}')
        self.alphabet = pl_task.alphabet
        self.data_processor = DataProcessor()

        self.cfg.generator = utils.config.merge_config(
            pl_task.hparams.generator, self.cfg.generator
        )
        # print(f'[DEBUG - designer.py] Initializing self.generator with {self.cfg.generator} and self.alphabet')
        self.generator = IterativeRefinementGenerator(
            alphabet=self.alphabet, 
            **self.cfg.generator
        )

        self._structure: dict = None
        self._predictions: list = None

    def print_config(self, print_exp_cfg=False):
        if print_exp_cfg:
            print(f"======= Experiment Config =======")
            print(OmegaConf.to_yaml(self.exp_cfg.resolve()))        

        print(f"======= Designer Config =======")
        print(OmegaConf.to_yaml(self.cfg))        

    def _cuda(self):
        assert torch.cuda.is_available()
        self.model = self.model.cuda()
        self._device = next(self.model.parameters()).device

    def reset(self):
        self._structure = None
        self._predictions = None

    def set_structure(
            self, 
            pdb_path, 
            chain_list=[],
            masked_chain_list=None,
            verbose=False
        ):
        pdb_id = Path(pdb_path).stem

        print(f'loading backbone structure from {pdb_path}.')
        # coords, native_seq = io.load_coords(pdb_path, None)
        self._structure = self.data_processor.parse_PDB(
            pdb_path, 
            input_chain_list=chain_list, # A list like chain_list=['B', 'A']
            masked_chain_list=masked_chain_list # A list like masked_chain_list=['B']
        )
        # print(f'self._structure={self._structure}')
        '''
        self._structure will have the following keys:
                { 'seq_chain_A': sequence of chain A,
                  'coords_chain_A': coordinates of chain A,
                  'seq_chain_B': sequence of chain B,
                  'coords_chain_B': coordinates of chain B,
                  'name': name of the biounit,
                  'num_of_chains': 2,
                  'seq': concatenated sequence of all chains,
                  'coords': concatenated coordinates of all chains,
                  'masked_list': ['B'] (masked chain),
                  'visible_list': ['A'] (visible chains),
                }
        '''

        # self._structure = {
        #     'name': pdb_id,
        #     'coords': coords, 
        #     'seq': native_seq
        # }
        if verbose: return self._structure

    def _featurize(self):
        batch = self.alphabet.featurize(raw_batch=[self._structure])

        if self.cfg.cuda:
            batch = utils.recursive_to(batch, self._device)

        prev_tokens, prev_token_mask = _full_mask(
            batch['tokens'], batch['coord_mask'], self.alphabet
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)
        return batch

    def generate(self, generator_args={}, need_attn_weights=False):
        batch = self._featurize()
        print(batch.keys())

        outputs = self.generator.generate(
            model=self.model, 
            batch=batch,
            need_attn_weights=need_attn_weights,
            **generator_args
        )

        output_tokens = outputs[0]
        output_tokens = self.alphabet.decode(output_tokens, remove_special=True)

        self._predictions = GenOut(
            output_tokens=output_tokens, 
            output_scores=outputs[1],
            attentions=outputs[2] if need_attn_weights else None
        )
        return self._predictions

    def calculate_metrics(self):
        native_seq = self._structure['seq']

        for prediction in self._predictions.output_tokens:
            rec = np.mean([(a==b) for a, b in zip(native_seq, prediction)])
            print(f"prediction: {prediction}")
            print(f"recovery: {rec}")
            print()


    def export_attention(self, saveto, layer=-1, average_heads=False):
        assert self._predictions is not None
        attentions = self._predictions.attentions[-1]

        from bertviz import model_view, head_view

        tokens = self.alphabet.decode(attentions['input'][None], return_as='list')[0]
        attns = attentions['attn_weights'].split(1, dim=0)
        num_layers = len(attns)

        if layer != 'all':
            layer = (num_layers + layer) % num_layers
            saveto = f"{saveto}_l{layer}"
            attns = [attns[layer]]
        if average_heads:
            attns = [attn.mean(dim=1, keepdims=True) for attn in attns]

        html = model_view(attns, tokens, html_action='return')

        with open(saveto + '.html', 'w') as f:
            f.write(html.data)

    def inpaint(self, start_ids, end_ids, generator_args={}, need_attn_weights=False):
        # print(f'[DEBUG - designer.py] Inpainting with start_ids={start_ids} and end_ids={end_ids}')
        # print(f'[DEBUG - designer.py] self._structure={self._structure}')
        batch = self.alphabet.featurize(raw_batch=[self._structure]) # This will use the multichain featurizer
        # print(f'[DEGUG - designer.py] Featurized batch={batch}')
        if self.cfg.cuda:
            print('Using GPU')
            batch = utils.recursive_to(batch, self._device)
        prev_tokens = batch['tokens'].clone()
        # print(f'[DEBUG - designer.py] Original tokens before masking: {self.alphabet.decode(prev_tokens, remove_special=True)}')

        # Print the segments we're going to mask
        print("\n===== CDR REGIONS TO BE MASKED =====")
        for i, (sid, eid) in enumerate(zip(start_ids, end_ids)):
            segment = self.alphabet.decode(prev_tokens[..., sid:eid+1], remove_special=True)
            print(f"CDR Region {i+1} [{sid}:{eid+1}] (length: {eid-sid+1}): {segment}")
            prev_tokens[..., sid:eid+1] = self.alphabet.mask_idx
        print("====================================\n")

        # print(f'[DEBUG - designer.py] Tokens after masking: {self.alphabet.decode(prev_tokens, remove_special=True)}')
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        # Print masking statistics
        total_tokens = batch['prev_token_mask'].numel()
        masked_tokens = batch['prev_token_mask'].sum().item()
        print(f"Total sequence length: {total_tokens}")
        print(f"Number of masked tokens: {masked_tokens} ({masked_tokens/total_tokens*100:.2f}% of sequence)")
        # print(f'[DEBUG - designer.py] Mask shape: {batch["prev_token_mask"].shape}, Sum of masked tokens: {batch["prev_token_mask"].sum().item()}')

        # print('[DEBUG - designer.py] Generating with the following featurized batch:')
        # print(f'Keys in batch: {batch.keys()}')
        # print(f'prev_token_mask shape: {batch["prev_token_mask"].shape}, sum: {batch["prev_token_mask"].sum().item()}')
        # Print strategy information
        strategy = generator_args.get('strategy', self.cfg.generator.strategy)
        print(f"Using generation strategy: {strategy}")
        if strategy == 'cdr_mask_predict':
            print("CDR-specific masking: Only tokens in CDR regions will be iteratively remasked based on confidence scores")
        
        # Use replace_visible_tokens=True to preserve non-masked tokens
        outputs = self.generator.generate(
            model=self.model,
            batch=batch,
            need_attn_weights=need_attn_weights,
            replace_visible_tokens=True,  # Ensures non-masked regions remain fixed
            **generator_args
        )
        
        output_tokens = outputs[0]

        original_segments = []
        designed_segments = []
        for sid, eid in zip(start_ids, end_ids): 
            original_segment = self.alphabet.decode(
                batch['tokens'][..., sid:eid+1].clone(), remove_special=False)
            original_segments.append(original_segment)

            designed_segment = self.alphabet.decode(
                output_tokens[..., sid:eid+1].clone(), remove_special=False)
            designed_segments.append(designed_segment)
            # print(f'[DEBUG - designer.py] Original segment [{sid}:{eid+1}]: {original_segment}')
            # print(f'[DEBUG - designer.py] Designed segment [{sid}:{eid+1}]: {designed_segment}')

        output_tokens = self.alphabet.decode(output_tokens, remove_special=True)
        self._predictions = GenOut(
            output_tokens=output_tokens, 
            output_scores=outputs[1],
            attentions=outputs[2] if need_attn_weights else None
        )
        return self._predictions, original_segments, designed_segments