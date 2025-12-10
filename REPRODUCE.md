
## Reproducing our results

### 1) Setup environment

```bash
git clone https://github.com/heidelberg-hepml/tagger-quantization
cd tagger-quantization
```

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2) Collect datasets

```bash
python data/collect_data.py toptagging
```


### 3) Training taggers

<!-- Table 1 (1M taggers)
```bash
python run.py -cp config model=tag_top_transformer training=top_transformer
python run.py -cp config model=tag_top_transformer training=top_transformer model.use_amp=true
python run.py -cp config model=tag_top_transformer training=top_transformer model.use_amp=true inputquant.use=true
python run.py -cp config model=tag_top_transformer training=top_transformer weightquant.use=true
python run.py -cp config model=tag_top_transformer training=top_transformer model.use_amp=true weightquant.use=true
python run.py -cp config model=tag_top_transformer training=top_transformer model.use_amp=true inputquant.use=true weightquant.use=true

# repeat different quantization levels for each of these
python run.py -cp config model=tag_lgatr training=top_lgatr
python run.py -cp config model=tag_lotr training=top_lotr
python run.py -cp config model=tag_ParT training=top_transformer
python run.py -cp config model=tag_top_transformer training=top_transformer model/framesnet=learnedpd
```

Table 2 (1k taggers)
```bash
python run.py -cp config model=tag_transformer_1k training=top_1k
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true inputquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_1k weightquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true weightquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true inputquant.use=true weightquant.use=true

# repeat different quantization levels for each of these
python run.py -cp config model=tag_lgatr_1k training=top_1k
python run.py -cp config model=tag_lotr_1k training=top_1k
python run.py -cp config model=tag_ParT_1k training=top_1k
python run.py -cp config model=tag_transformer_1k training=top_1k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k
``` -->

Figure 3: Scaling
```bash
python run.py -cp config model=tag_transformer_1k training=top_1k
python run.py -cp config model=tag_transformer_10k training=top_10k
python run.py -cp config model=tag_transformer_100k training=top_100k
python run.py -cp config model=tag_transformer training=top_transformer
#Repeat for each network
python run.py -cp config model=tag_lotr_10k training=top_10k
python run.py -cp config model=tag_part_10k training=top_10k
python run.py -cp config model=tag_transformer_10k training=top_10k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_10k
```

Table 6: Landscape of fp8+QAT top tagger
```bash
python run.py -cp config model=tag_transformer training=top_transformer training.scheduler=CosineAnnealingLR weightquant.use=true inputquant.use=true model.use_amp=true
python run.py -cp config model=tag_transformer model/framesnet=learnedpd model/framesnet/equivectors=equimlp training=top_transformer training.scheduler=CosineAnnealingLR weightquant.use=true inputquant.use=true model.use_amp=true
python run.py -cp config model=tag_lgatr training=top_lgatr weightquant.use=true inputquant.use=true model.use_amp=true
python run.py -cp config model=tag_lotr training=top_lotr weightquant.use=true inputquant.use=true model.use_amp=true
python run.py -cp config model=tag_ParT training=top_transformer training.scheduler=CosineAnnealingLR weightquant.use=true inputquant.use=true model.use_amp=true
```

Figure 7
```bash
python run.py -cp config model=tag_transformer training=top_transformer
python run.py -cp config model=tag_transformer training=top_transformer model.use_amp=true
python run.py -cp config model=tag_transformer training=top_transformer model.use_amp=true inputquant.use=true
python run.py -cp config model=tag_transformer training=top_transformer training.scheduler=CosineAnnealingLR model.use_amp=true inputquant.use=true
# Repeat for each network, with CosineAnnealingLR when using top_transformer training config
```

Figure 8
```bash
python run.py -cp config model=tag_transformer_1k training=top_1k
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true inputquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true inputquant.use=true weightquant.use=true
# Repeat for each network
```

### 4) Estimating computational cost

The cost models in `cost_estimate/` can be used to estimate the energy consumption (cost measure on GPUs) as well as the number of bit-operations (cost measure on FPGAs).

The cost models are calibrated by comparing the corresponding FLOPs to FLOPs estimated with `torch.utils.flop_counter.FlopCounterMode`. Note that `FlopCounterMode` neglects a range of operations, especially when run on CPU (it neglects attention on CPU in some cases). For this reason, the two FLOPs estimates are not expected to match perfectly.

```bash
pytest tests/cost_estimate/test_flops.py -s
```

To reproduce the cost estimates used in the paper, run these commands. They produce `json` files in the same directory that store the information.

```bash
# energy cost estimate for large taggers (on GPU)
python cost_estimate/estimate_energy.py

# bitops cost for small taggers (on FPGAs)
python cost_estimate/estimate_bitops.py
```
