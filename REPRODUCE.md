
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
python run.py -cp config model=tag_top_transformer training=top_transformer use_amp=true
python run.py -cp config model=tag_top_transformer training=top_transformer use_amp=true inputquant.use=true
python run.py -cp config model=tag_top_transformer training=top_transformer weightquant.use=true
python run.py -cp config model=tag_top_transformer training=top_transformer use_amp=true weightquant.use=true
python run.py -cp config model=tag_top_transformer training=top_transformer use_amp=true inputquant.use=true weightquant.use=true

# repeat different quantization levels for each of these
python run.py -cp config model=tag_lgatr training=top_lgatr
python run.py -cp config model=tag_lotr training=top_lotr
python run.py -cp config model=tag_ParT training=top_lotr
python run.py -cp config model=tag_top_transformer training=top_transformer model/framesnet=learnedpd
```

Table 2 (1k taggers)
```bash
python run.py -cp config model=tag_transformer_1k training=top_small
python run.py -cp config model=tag_transformer_1k training=top_small use_amp=true
python run.py -cp config model=tag_transformer_1k training=top_small use_amp=true inputquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_small weightquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_small use_amp=true weightquant.use=true
python run.py -cp config model=tag_transformer_1k training=top_small use_amp=true inputquant.use=true weightquant.use=true

# repeat different quantization levels for each of these
python run.py -cp config model=tag_lgatr_1k training=top_small
python run.py -cp config model=tag_lotr_1k training=top_small
python run.py -cp config model=tag_ParT_1k training=top_small
python run.py -cp config model=tag_transformer_1k training=top_small model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k
```
