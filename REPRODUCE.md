
## Reproducing our results

This repo is based on https://github.com/heidelberg-hepml/lloca-experiments. For instructions on how to run our code, please have a look at that repo, in particular the REPRODUCE.md file.

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

Download [JetClass dataset](https://zenodo.org/records/6619768) and put its path in `config/jctagging.yaml` `data.data_dir`.

### 3) Training taggers

Table 1: Standard top tagging
```bash
# ParT new 'longer' vs literature config
python run.py -cp config model=tag_ParT training=top_ParT # literature config
python run.py -cp config model=tag_ParT training=top_slim # 'longer' config

# Standard L-GATr-slim
python run.py -cp config model=tag_slim training=top_slim

# Pretraining + finetuning L-GATr-slim
python run.py -cp config -cn jctagging model=tag_slim training=jc_ParT data.features=fourmomenta exp_name=pretrain run_name=slim-pretrain
python run.py -cp config -cn toptaggingft finetune.backbone_path=runs/pretrain/slim-pretrain training=top_slim
```

Table 2: Standard JetClass
```bash
python run.py -cp config -cn jctagging model=tag_slim training=jc_ParT
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd training=jc_ParT # updated compared to https://arxiv.org/abs/2508.14898, now using smaller Frames-Net
```

Table 3: Amplitude regression

Run this code in the [lloca-experiments repo](github.com/heidelberg-hepml/lloca-experiments).
```bash
python run.py -cp config -cn amplitudesxl model=amp_slim data.num_train_files=10
```

The figure in the right panel of table 3 is obtained by running similar commands for all other networks, and then extracting the amplitudes in the test dataset to create the plot.

Figure 2: Event generation

Run this code in the [lloca-experiments repo](github.com/heidelberg-hepml/lloca-experiments).
```bash
# scaling with training dataset size
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.5,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.2,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.05,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.02,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.005,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.002,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.0005,0.2,0.1]
python run.py -cp config -cn ttbar model=eg_slim data.train_test_val=[0.0002,0.2,0.1]

# scaling with multiplicity
python run.py -cp config -cn ttbar model=eg_slim data.n_jets=0
python run.py -cp config -cn ttbar model=eg_slim data.n_jets=1
python run.py -cp config -cn ttbar model=eg_slim data.n_jets=2
python run.py -cp config -cn ttbar model=eg_slim data.n_jets=3
python run.py -cp config -cn ttbar model=eg_slim data.n_jets=4
```

Figure 3: Scaling to small top taggers
```bash
python run.py -cp config model=tag_transformer_1k training=top_1k
python run.py -cp config model=tag_transformer_10k training=top_10k
python run.py -cp config model=tag_transformer_100k training=top_100k
python run.py -cp config model=tag_top_transformer training=top_transformer

python run.py -cp config model=tag_transformer_1k training=top_1k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k
python run.py -cp config model=tag_transformer_10k training=top_10k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_10k
python run.py -cp config model=tag_transformer_100k training=top_100k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_100k
python run.py -cp config model=tag_top_transformer training=top_transformer model/framesnet=learnedpd

python run.py -cp config model=tag_transformer_1k training=top_1k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k model.framesnet.is_global=true
python run.py -cp config model=tag_transformer_10k training=top_10k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_10k model.framesnet.is_global=true
python run.py -cp config model=tag_transformer_100k training=top_100k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_100k model.framesnet.is_global=true
python run.py -cp config model=tag_top_transformer training=top_transformer model/framesnet=learnedpd model.framesnet.is_global=true

python run.py -cp config model=tag_slim_1k training=top_1k
python run.py -cp config model=tag_slim_10k training=top_10k
python run.py -cp config model=tag_slim_100k training=top_100k
python run.py -cp config model=tag_slim training=top_slim

python run.py -cp config model=tag_ParT_10k training=top_10k
python run.py -cp config model=tag_ParT_100k training=top_100k
python run.py -cp config model=tag_ParT training=top_ParT

# Repeat with model=tag_X_deep_1k for right plot (including model=tag_ParT_deep_1k)
python run.py -cp config model=tag_transformer_deep_1k training=top_1k
# and so on...
```

Table 5 (landscape of fp8+QAT top taggers) and Figure 5 (energy consumption of large taggers)
```bash
python run.py -cp config model=tag_top_transformer training=top_transformer # fp32
python run.py -cp config model=tag_top_transformer training=top_transformer model.use_amp=true # fp16
python run.py -cp config model=tag_top_transformer training=top_transformer model.use_amp=true inputquant.use=true # fp8
python run.py -cp config model=tag_top_transformer training=top_transformer training.scheduler=CosineAnnealingLR model.use_amp=true inputquant.use=true weightquant.use=true # fp8+PARQ
python run.py -cp config model=tag_top_transformer training=top_transformer training.scheduler=CosineAnnealingLR model.use_amp=true inputquant.use=true weightquant.use=true weightquant.prox_map=hard # fp8+STE

# Repeat the five quantization levels for each network
# with training.scheduler=CosineAnnealingLR for tag_transformer and fp8+PARQ/STE
python run.py -cp config model=tag_ParT training=top_slim
python run.py -cp config model=tag_top_transformer model/framesnet=learnedpd training=top_transformer
python run.py -cp config model=tag_slim training=top_slim
```

Figure 6: Performance over bit operations for small taggers
```bash
python run.py -cp config model=tag_transformer_1k training=top_1k
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true
python run.py -cp config model=tag_transformer_1k training=top_1k model.use_amp=true inputquant.use=true

# Repeat the three quantization levels for the following networks
python run.py -cp config model=tag_transformer_1k training=top_1k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k
python run.py -cp config model=tag_transformer_1k training=top_1k model/framesnet=learnedpd model/framesnet/equivectors=equimlp_1k model.framesnet.is_global=true

# Repeat the three quantization levels for the three networks with less than 30 jet constituents
python run.py -cp config model=tag_transformer_1k training=top_1k data.max_particles=30
```

### 4) Estimating computational cost

The cost models in `cost_estimate/` can be used to estimate the energy consumption (cost measure on GPUs) as well as the number of bit-operations (cost measure on FPGAs).

The cost models are calibrated by comparing the corresponding FLOPs to FLOPs estimated with `torch.utils.flop_counter.FlopCounterMode`. Note that `FlopCounterMode` neglects a range of operations, especially when run on CPU (it neglects attention on CPU in some cases). For this reason, the two FLOPs estimates are not expected to match perfectly.

```bash
pytest tests/cost_estimate/test_flops.py -s # run on GPU
```

To reproduce the cost estimates used in the paper, run these commands. They produce `json` files in the same directory that store the information.

```bash
# energy cost estimate for 1M taggers
python cost_estimate/estimate_energy.py

# bitops cost for 1k taggers
python cost_estimate/estimate_bitops.py
```
