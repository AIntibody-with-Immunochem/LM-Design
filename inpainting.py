from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

# 1. instantialize designer
exp_path = "weights/lm_design_esm2_650m"
cfg = Cfg(
    cuda=True,
    generator=Cfg(
        max_iter=10,  # Increased from 5 to 10 to allow more refinement iterations
        strategy='cdr_mask_predict',  # ['denoise' | 'mask_predict' | 'cdr_mask_predict']
        temperature=0,
        eval_sc=False,
    )
)
designer = Designer(experiment_path=exp_path, cfg=cfg)

pdb_path = "input/anti_ClfA_design_16.pdb"


# designer.set_structure(pdb_path)

# # 3. generate sequence from the given structure
# designer.generate()

# # 4. calculate evaluation metrics
# designer.calculate_metrics()




# PDB Layout (4KRL):
# B - VHH binder chain
# A - target antigen chain
# print('[DEBUG - inpainting.py] Setting structure...')
# designer.set_structure(pdb_path, chain_list=['B', 'A'], 
#                        masked_chain_list=['B'])  
designer.set_structure(pdb_path, masked_chain_list=['H'], # which chains to predict while the remaining chains serve as conditioning
                       chain_list=['H', 'T']) # Put binder chain first

'''
self_structure wiill set self._structure to the following dictionary:

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

# 2. Set CDRs
'''
4KRL:
CDR-H1: 26-32 (Length: 7)
    0-based index: 25-31
CDR-H2: 52-57 (Length: 6)
    0-based index: 51-56
CDR-H3: 99-113 (Length: 15)
    0-based index: 98-112
    
anti-ClfA:
CDR-H1: 26-33 (Length: 8)
CDR-H2: 51-57 (Length: 7)
CDR-H3: 96-111 (Length: 16)
'''
start_ids = [26, 51, 96]
end_ids = [33, 57, 111]
binder_len = 122
num_seqs = 10

# 3. inpaint
# Reduce the number of sequences for debugging
debug_seqs = 2  # Just generate 2 sequences for easier debug analysis
for i in range(debug_seqs):
    print('\n\n========== INPAINTING SEQUENCE #{} =========='.format(i+1))
    out, ori_seg, designed_seg = designer.inpaint(
        start_ids=start_ids, end_ids=end_ids,
        generator_args={'temperature': 0.20, 'strategy': 'cdr_mask_predict'}
    )
    print('Designed Segments:')
    print(designed_seg)
    print('Designed Sequence:')
    print(out.output_tokens[0][:binder_len])
print('\nOriginal Segments:')
print(ori_seg)

