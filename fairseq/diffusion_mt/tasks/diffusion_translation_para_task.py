# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguageParaDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
import itertools
import os
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
)

Q_SAMPLE_CHOICES = ChoiceEnum(["default", "coupled", "multi-sample", "multi-step"])
EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)

def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        #logger.info(src_dataset)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    # logger.info(len(src_datasets))--> 1
    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
    # logger.info(prepend_bos)-> True
    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguageParaDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        
    )

@dataclass
class DiffusionTranslationConfig(TranslationConfig):
    # Diffusion arch. arguments
    timestep_emb_type: str = field(
        default="sinusoidal", 
        metadata={"help": "Type of the timestep embeddings"}
    )
    num_diffusion_timesteps: int = field(
        default=20,
        metadata={"help": "Number of total diffusion timesteps"}
    )
    diffusion_type: str = field(
        default="absorbing",
        metadata={"help": "The type of the discrete diffusion process."}
    )
    noise_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The noise schedule type used in Multinomial Diffusion Process."}
    )
    noise_distribution: str = field(
        default="uniform",
        metadata={"help": "The noise distribution (either unigram or uniform) in multinomial diffusion processes."}
    )
    scale_attn_masking: bool = field(
        default=False,
        metadata={"help": "Whether to attenuate the effect of noisy tokens in attention."}
    )

    # constructing q samples arguments
    q_sample_mode: Q_SAMPLE_CHOICES = field(
        default="default", 
        metadata={"help": "Type of the time sampler"}
    )
    time_sampler_type: str = field(
        default="uniform", 
        metadata={"help": "Type of the time sampler"}
    )
    not_diffusing_special_sym: bool = field(
        default=False,
        metadata={"help": "whether the special symbols are masked or not in sampling q(x_t | x_0)."}
    )

    # training loss arguments
    reweighting_type: str = field(
        default="linear",
        metadata={"help": "The type of reweighting for the cross-entropy loss function."}
    )
    lambda_direct_xentropy: float = field(
        default=-1.0, 
        metadata={"help": "Coefficient of an auxliary cross_entropy loss"}
    )
    decoder_loss_type: str = field(
        default="orig", 
        metadata={"help": "The definition of log p(x_0 | x_1)."}
    )

    # decoding specifics
    argmax_decoding: bool = field(
        default=False,
        metadata={"help": "Whether use deterministic decoding or not."}
    )
    decoding_time_difference: int = field(
        default=0,
        metadata={"help": "Asymmetric time interval technique proposed in Bit Diffusion."}
    )
    temperature_annealing: bool = field(
        default=False,
        metadata={"help": "Anneal the temperature during decoding steps."}
    )
    return_all_cands: bool = field(
        default=False,
        metadata={"help": "Return the best generated sent or all sents."}
    )
    decoding_strategy: str = field(
        default="default",
        metadata={"help": "The Skip-step schedule during decoding."}
    )
    beam_within_length: int = field(
        default=1,
        metadata={"help": "The beam size within each length."}
    )

    # generation args workaround
    load_ema_weights: bool = field(
        default=False,
        metadata={"help": "Load EMA model weights for generation inference."}
    )

@register_task("diffusion_para_translation", dataclass=DiffusionTranslationConfig)
class DiffusionParaTranslationTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Discrete Diffusion Models
    """

    cfg: DiffusionTranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )
        
    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from diffusion_mt.diffusion_generator import DiffusionGenerator

        task_args = unused.get("task_args", None)
        if task_args is not None:
            decoding_time_difference = getattr(task_args, "decoding_time_difference", 0)
            argmax_decoding = getattr(task_args, "argmax_decoding", False)
            temperature_annealing = getattr(task_args, "temperature_annealing", False)
            decoding_strategy = getattr(task_args, "decoding_strategy", "default")
            return_all_cands = getattr(task_args, "return_all_cands", False)
            beam_within_length = getattr(task_args, "beam_within_length", 1)
        else:
            decoding_time_difference = getattr(args, "decoding_time_difference", 0)
            argmax_decoding = getattr(args, "argmax_decoding", False)
            temperature_annealing = getattr(args, "temperature_annealing", False)
            decoding_strategy = getattr(args, "decoding_strategy", "default")
            return_all_cands = getattr(args, "return_all_cands", False)
            beam_within_length = getattr(args, "beam_within_length", 1)

        decoder_option_args = {
            "eos_penalty": getattr(args, "iter_decode_eos_penalty", 0.0),
            "max_ratio": getattr(args, "iter_decode_max_ratio", 2),
            "decoding_format": getattr(args, "decoding_format", None),
            "temperature": getattr(args, "temperature", 1.0),
            "decoding_time_difference": decoding_time_difference,
            "argmax_decoding": argmax_decoding,
            "temperature_annealing": temperature_annealing,
            "decoding_strategy": decoding_strategy,
        }
        return DiffusionGenerator(
            self.target_dictionary,
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            beam_within_length=beam_within_length,
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
            decoder_options=decoder_option_args,
            in_pretrain=getattr(args, "in_pretrain",False),
            return_all_cands=return_all_cands,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the diffusion_translation task is not supported"
            )

        return LanguageParaDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
            if self.cfg.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode( 
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], force=True, tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs], force=True)
