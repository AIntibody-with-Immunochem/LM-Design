from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

# 1. instantialize designer
exp_path = "/root/research/projects/ByProt/run/logs/fixedbb/cath_4.2/lm_design_esm2_650m"
cfg = Cfg(
    cuda=True,
    generator=Cfg(
        max_iter=5,
        strategy='denoise', 
        temperature=0,
        eval_sc=False,  
    )
)
designer = Designer(experiment_path=exp_path, cfg=cfg)

# 2. load structure from pdb file
pdb_path = "/root/research/projects/ByProt/data/3uat_variants/3uat_GK.pdb"
designer.set_structure(pdb_path)

# 3. generate sequence from the given structure
designer.generate()
# you can override generator arguments by passing generator_args, e.g.,
designer.generate(
    generator_args={
        'max_iter': 5, 
        'temperature': 0.1,
    }
)

# 4. calculate evaluation metircs
designer.calculate_metrics()
## prediction: LNYTRPVIILGPFKDRMNDDLLSEMPDKFGSCVPHTTRPKREYEIDGRDYHFVSSREEMEKDIQNHEFIEAGEYNDNLYGTSIESVREVAMEGKHCILDVSGNAIQRLIKADLYPIAIFIRPRSVENVREMNKRLTEEQAKEIFERAQELEEEFMKYFTAIVEGDTFEEIYNQVKSIIEEESG
## recovery: 0.7595628415300546
