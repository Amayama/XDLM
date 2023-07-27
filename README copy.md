# XDLM: Cross-lingual Diffusion Language Model for Machine Translation

This repository contains the official implementation of paper [XDLM: Cross-lingual Diffusion Language Model for Machine Translation](https://arxiv.org/abs/2307.13560)

## Dependencies

The codebase is implemented with [FairSeq](https://github.com/facebookresearch/fairseq). To install the dependencies, run (recommended in a [virtual environment](https://docs.python.org/3/library/venv.html)) the following commands:

```bash
pip install -r requirements.txt

# install our package of discrete diffusion models
pip install -e discrete_diffusion
# install our fork of fairseq
cd fairseq
python3 setup.py build develop
cd ..
```

> **Note**
> The environment is tested with Python 3.8.10, PyTorch 1.10.0/1.12.0, and CUDA 11.3.
> Also note our fork of fairseq modifies several files in the original codebase; using more recent versions of fairseq might lead to unexpected dependency conflicts.

## Preprocess of source

### pretrain stage-para dataset

For the preprocess of opus dataset, we use [script](xdiff/data_process/get-para-data.sh) to download correspond language pairs as following instruction.

```
# Download and tokenize parallel data in 'data/wiki/para/en-zh.{en,zh}.{train,valid,test}'
./get-data-para.sh en-zh &
```

For the using of BPE tools, we use the tools introduced in [here](https://github.com/facebookresearch/XLM/tree/main/tools), use [script](xdiff/data_process/preprocess-para.sh) to obtain the BPE code and processed dataset.

Use following instruction. Also, the processed data-bin/para data is available [here](https://drive.google.com/file/d/154WQ6LS_qlbPCUenJR0PBitz6wepIKSR/view?usp=sharing).

```

TEXT=para
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.en-de --validpref $TEXT/valid.en-de --testpref $TEXT/test.en-de \
    --destdir data-bin/para --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20


```

### Finetune stage-wmt14/IWSLT14 dataset

We use [Huggingface](https://huggingface.co/) to obtain the origin data source from [here](https://huggingface.co/datasets/wmt14).

For finetune datasets, we use the BPE codes obtained from opus dataset, and generate the processed dataset by the applyBPE operation used in script.

We use following instructions to process and binarize the dataset for finetuning. Also, the processed data-bin/wmt14-ende data is available [here](https://drive.google.com/file/d/1TxAL9KOdR1LHtUmQHPq6fG_xUt3oMl6x/view?usp=sharing).

```

TEXT=wmt14_ende/IWSLT
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.en-de --validpref $TEXT/valid.en-de --testpref $TEXT/test.en-de \
    --destdir data-bin/wmt14_ende --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

```

Don't forget to replace the `dict.en.txt` and `dict.de.txt` in the downloaded folder as the vocab built in the previous stage before apply `fairseq-preprocess`

## Pretrain and Finetune stage

As we introduced in the paper, we use the reparameterized multinomial diffusion model for both pretrain process and finetune process.

### Pretrain code

```
bash experiments/mt_train.sh -m reparam-multinomial -d para -s default -e True --not-diffusing-special-sym --q-sample-mode coupled --store-ema --label-smoothing 0.1 --reweighting-type linear
```

### Finetune Code

Before we finetune on the pretrained model, please move the checkpoints to the potential checkpoint position and rename it as a checkpoint_last.pt

```
bash experiments/mt_train_finetune.sh -m reparam-multinomial -d wmt14/IWSLT -s default -e True --not-diffusing-special-sym --q-sample-mode coupled --store-ema --label-smoothing 0.1 --reweighting-type linear
```

### Decoding Strategies

#### Vanilla Sampling Scheme

By passing `--decoding-strategy default`, the vanilla sampling scheme (specific to each discrete diffusion process) is used.

#### Improved Sampling with Reparameterization

A more advanced decoding approach can be invoked by passing `--decoding-strategy reparam-<conditioning-of-v>-<topk_mode>-<schedule>`. This approach is based on the proposed reparameterization in our paper and allows for more effective decoding procedures. The options specify the decoding algorithm via

- `<conditioning-of-v>`: `uncond` or `cond` (default `uncond`): whether to generate the routing variable $v_t$ in a conditional or unconditional manner;
- `<topk_mode>`: `stochastic<float>` or `deterministic` (default `deterministic`): whether to use stochastic or deterministic top-$k$ selection. The float value in `stochastic<float>` specifies the degree of randomness in the stochastic top-$k$ selection;
- `<schedule>`: `linear` or `cosine` (default `cosine`): the schedule for $k$ during our denoising procedure, which is used to control the number of top-$k$ tokens to be denoised for the next decoding step.

See the [implementation](./discrete_diffusion/discrete_diffusions/discrete_diffusion_base.py#L130) for more details about the options.

## Machine Translation

### Data Preprocessing

Please see the scripts below for details.

> **Note**
>
> - Note that all tasks considered in this work operate on the original data and do **not** adopt Knowledge Distillation (KD).

#### Training

We first get into the `fairseq` folder and then run the following commands to train the models.

```bash
######## training scripts for IWSLT'14 , WMT'14, and WMT'16 
# first cd to fairseq
# we use 1 GPU for IWSLT'14, 4 GPUs for WMT'14 and 2 GPUs for WMT'16 datasets respectively.
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_train.sh -m absorbing -d <iwslt/wmt14/wmt16> -s default -e True --store-ema --label-smoothing 0.1
CUDA_VISIBLE_DEVICES=1 bash experiments/mt_train.sh -m multinomial -d <iwslt/wmt14/wmt16> -s default -e True --not-diffusing-special-sym --store-ema --label-smoothing 0.0
CUDA_VISIBLE_DEVICES=2 bash experiments/mt_train.sh -m reparam-absorbing -d <iwslt/wmt14/wmt16> -s default -e True --q-sample-mode coupled  --store-ema --label-smoothing 0.1 --reweighting-type linear
CUDA_VISIBLE_DEVICES=3 bash experiments/mt_train.sh -m reparam-multinomial -d <iwslt/wmt14/wmt16> -s default -e True --not-diffusing-special-sym --q-sample-mode coupled --store-ema --label-smoothing 0.1 --reweighting-type linear
```

> **Note**
>
> - `-s <str>` is used to specify the name of the experiment.
> - We could pass custom arguments that might be specific to training by appending them after `-e True`.

### Generation & Evaluation

The evaluation pipeline is handled by `experiments/mt_generate.sh`. The script will generate the translation results and evaluate the BLEU score.

```bash
########### IWLS'14 and WMT'14 datasets
# we recommend putting each checkpoint into a separate folder
# since the script will put the decoded results into a file under the same folder of each checkpoint.
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d <iwslt/wmt14> 
```

Arguments:

- `-a`: whether to average multiple checkpoints
- `-c`: indicates the location of the checkpoint.
  If `-a false` (not to average checkpoints), pass the checkpoint **path**;
  if `-a true`, pass the **directory** that stores multiple checkpoints at different training steps for averaging.
- `-d`: the dataset name

### Trained Model Checkpoints

We also provide the checkpoints of our trained models.

| Dataset  | Model               |                                                              Checkpoint link                                                              |
| -------- | ------------------- | :----------------------------------------------------------------------------------------------------------------------------------------: |
| IWSLT'14 | Multinomial         | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EpAzao9L5XBMsef5LNZ1iXkB36Mp9V2gQGOwbopgPaOTVA?e=OraA81) |
| IWSLT'14 | Absorbing           | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Eg1tqijPqkpNvc0Lai-BDE0Btc8L4UIJ-7oedCp4MXDPKw?e=liuASC) |
| IWSLT'14 | Reparam-multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EmCnWDgoj8JKmji1QE8UlkMB-3ow1aI8Bdo78-C7LqU_hA?e=DNahYn) |
| IWSLT'14 | Reparam-absorbing   | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EmvmYZemCIRMsKQF-GNitzQB1lRUYj5MSow9jyxHZ4BCUg?e=nS81rB) |
| WMT'14   | Multinomial         | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Ehgx0Ur0fbdJgY0zreg4KbABrN21txHM-sisbR9xZ6unDQ?e=T1vnJL) |
| WMT'14   | Absorbing           | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EtO0Hft6GmhKogahr4V1hnQB4Odt5MUcjSUXawg_lH_0wg?e=Ikzs3R) |
| WMT'14   | Reparam-multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EtfgIjc9g2tEh3F9IpcvFoUBmIkcihy_tpVezr845fEDtQ?e=uTYJYF) |
| WMT'14   | Reparam-absorbing   | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EniOmBTtL2dDtk1GNBw-kg4BsJ3SWTGmGASNdjRjSCP27w?e=Ona4qx) |
| WMT'16   | Multinomial         | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EiBNFip8De5Nk-kimmyQ3UYBftUH3Cz74RsiA9IfoIryBQ?e=tzswtp) |
| WMT'16   | Absorbing           | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EiFkp1Ros4VCsl4w-Feez7oB_h2zLEV61dHwsaFGxk7ioQ?e=96xT6h) |
| WMT'16   | Reparam-multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Em4byDij7zJIl1SY6nIcVeABbAEQZvsb1O8LdlS4i6t92A?e=0QQZaA) |
| WMT'16   | Reparam-absorbing   | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Ep5D3LYr7FJLiWOrPbm3T3YBWtloPcdlNOmh5k9nM6CuzA?e=7pC43S) |

## Citation

```bibtex
@misc{chen2023xdlm,
      title={XDLM: Cross-lingual Diffusion Language Model for Machine Translation}, 
      author={Linyao Chen and Aosong Feng and Boming Yang and Zihui Li},
      year={2023},
      eprint={2307.13560},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
