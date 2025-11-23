
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

Table 1 (1M taggers)
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
python run.py -cp config model=tag_transformer_1k training=top_small
python run.py -cp config model=tag_transformer_1k training=top_small model.use_amp=true
python run.py -cp config model=tag_transformer_1k training=top_small model.use_amp=true inputquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_small weightquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_small model.use_amp=true weightquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_small model.use_amp=true inputquant.use=true weightquant.use=true

# repeat different quantization levels for each of these
python run.py -cp config model=tag_lgatr_1k training=top_small
python run.py -cp config model=tag_lotr_1k training=top_small
python run.py -cp config model=tag_ParT_1k training=top_small
python run.py -cp config model=tag_transformer_1k training=top_small model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k
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
