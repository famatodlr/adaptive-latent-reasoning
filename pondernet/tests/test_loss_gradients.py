"""Regression tests for gradient flow through the PonderNet loss (exp-10 root cause).

WHAT WENT WRONG
---------------
`CODI.forward` assembles the total loss twice. The first assembly is graph-connected;
the components are then detached so the returned dict carries plain numbers for logging
(this is upstream CODI/SIM-CoT behaviour and is safe there, because upstream has finished
building `loss` before detaching). The PonderNet branch, however, *rebuilds* `loss` after
that point - it substitutes `l_pondernet` for the fixed-K `ce_loss_total`. It used to read
the already-detached names, so the effective objective silently became

    L = L_ponder + gamma * KL_geom

with CODI's self-distillation (`L_distill`, `L_ref`) and SIM-CoT's step-level supervision
(`L_step`) contributing exactly zero gradient. The auxiliary decoder, whose only loss is
`L_step`, therefore never trained in any run from experiment 02 to 11 and stayed at its
vanilla-GPT-2 initialisation. See docs/exp10-diagnosis.md section 4.0.

WHY THIS MATTERS (per the source papers)
----------------------------------------
* SIM-CoT (arXiv 2509.20317, section 3.5): "Gradients from L_step propagate through the
  decoder into the latent representations z_1:K and further into the LLM, shaping the
  hidden states to encode step-level reasoning." Gradient flow into the latents IS the
  mechanism; without it the auxiliary decoder is decorative.
* CODI (arXiv 2502.21074, Table 2): removing the distillation loss drops GSM8k-Aug
  accuracy from 43.7% to 24.5% at GPT-2 scale.

WHY THE ORIGINAL VERIFICATION MISSED IT
---------------------------------------
The exp-10 write-up claims the decoder was verified to train because "149/149 decoder.*
tensors differ from vanilla GPT-2 after 1 epoch". That test used `!=`, which bf16 storage
round-trip noise (~0.002 relative) already satisfies. These tests therefore assert on
gradient MAGNITUDE, never on mere inequality.

These tests run on CPU in float32 and use a tiny synthetic batch, so they are fast and need
no GPU and no dataset.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from peft import LoraConfig, TaskType

from src.model import CODI, ModelArguments, TrainingArguments, freeze_model

pytestmark = pytest.mark.skipif(
    os.environ.get("ALR_SKIP_MODEL_TESTS") == "1",
    reason="model-level tests need the gpt2 weights available locally",
)

GROUPS = {
    "backbone": lambda n: n.startswith("codi.") and "lora_" not in n,
    "lora": lambda n: "lora_" in n,
    "prj": lambda n: n.startswith("prj"),
    "decoder": lambda n: n.startswith("decoder."),
    "halt_head": lambda n: "halt_head" in n,
}


def _build_model(pondernet: bool, max_latent_steps: int = 3):
    model_args = ModelArguments(
        model_name_or_path="gpt2", lora_r=8, lora_alpha=32, full_precision=True,
        use_decoder=True, decoder_path="gpt2", train=True,
    )
    training_args = TrainingArguments(
        output_dir="/tmp/alr-test", bf16=False, use_lora=True, use_prj=True,
        prj_dim=64, prj_dropout=0.0, max_latent_steps=max_latent_steps,
        pondernet=pondernet, pondernet_halt_bias_init=-2.0, pondernet_beta=1.0,
        pondernet_gamma=0.10, pondernet_adaptive_prior=False, pondernet_geom_mean=3.0,
        remove_eos=True, print_loss=False, distill_loss_div_std=True, report_to=[],
    )
    # lora_dropout=0 so a forward pass is deterministic: the beta-scaling test below
    # compares two backward passes and must not have dropout noise between them.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
        lora_dropout=0.0, target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )
    torch.manual_seed(0)
    model = CODI(model_args, training_args, lora_config)
    # exp-10's scope (`full_dec`): every parameter trainable, decoder included.
    freeze_model(model)
    for _, p in model.named_parameters():
        p.requires_grad = True
    return model.float()


def _synthetic_batch(model):
    """A hand-built batch in the shape `forward` expects.

    Token ids are arbitrary but must contain the '<<'/'>>' expression markers that
    `get_steps` looks for (16791/9959 open, 4211 close) so the auxiliary-decoder branch
    produces a non-empty L_step - otherwise the test would pass vacuously.
    """
    pad = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else 50256
    eos = model.tokenizer.eos_token_id
    # ref sequence: [question][<<expr>>][<<expr>>][answer]
    ref = torch.tensor([
        [15496, 995, 16791, 17, 10, 18, 28, 20, 4211, 16791, 20, 9, 17, 28, 940, 4211,
         464, 3280, 318, 25, 838, eos],
        [15496, 612, 16791, 19, 10, 19, 28, 23, 4211, 16791, 23, 9, 17, 28, 1433, 4211,
         464, 3280, 318, 25, 1467, eos],
    ], dtype=torch.long)
    ref_labels = ref.clone()
    ref_labels[:, :2] = -100

    enc = torch.tensor([[15496, 995, model.bot_id],
                        [15496, 612, model.bot_id]], dtype=torch.long)
    dec = torch.tensor([[model.eot_id, 464, 3280, 318, 25, 838, eos],
                        [model.eot_id, 464, 3280, 318, 25, 1467, eos]], dtype=torch.long)
    labels = dec.clone()

    # position of the token right after "The answer is:" (index 5 in dec, 20 in ref)
    return dict(
        encoder_input_ids=enc,
        decoder_input_ids=dec,
        ref_input_ids=ref,
        labels=labels,
        encoder_attention_mask=enc.ne(pad),
        ref_answer_position=torch.tensor([20, 20], dtype=torch.long),
        model_answer_position=torch.tensor([5, 5], dtype=torch.long),
        ref_attention_mask=ref.ne(pad),
        ref_labels=ref_labels,
    )


def _grad_magnitude(model, predicate):
    total = 0.0
    for name, p in model.named_parameters():
        if predicate(name) and p.requires_grad and p.grad is not None:
            total += p.grad.float().abs().sum().item()
    return total


def _backward(model, batch):
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out["loss"].backward()
    return out


def test_pondernet_propagates_gradient_to_auxiliary_decoder():
    """THE regression test for the exp-10 root cause.

    In PonderNet mode the auxiliary decoder must receive gradient, because L_step is the
    only loss touching it and SIM-CoT's whole contribution is that L_step shapes the
    latents. Before the fix this was exactly 0.0 on all 148 decoder tensors.
    """
    model = _build_model(pondernet=True)
    _backward(model, _synthetic_batch(model))

    decoder_grad = _grad_magnitude(model, GROUPS["decoder"])
    assert decoder_grad > 1e-6, (
        "auxiliary decoder received no gradient in PonderNet mode - the PonderNet loss "
        "is being rebuilt from detached components again (exp-10 root cause). "
        f"sum|grad| = {decoder_grad}"
    )


def test_pondernet_all_trainable_groups_receive_gradient():
    """Every group that `full_dec` marks trainable must actually train."""
    model = _build_model(pondernet=True)
    _backward(model, _synthetic_batch(model))

    for group, predicate in GROUPS.items():
        magnitude = _grad_magnitude(model, predicate)
        assert magnitude > 1e-6, f"group {group!r} received no gradient (sum|grad|={magnitude})"


def test_pondernet_decoder_gradient_scales_with_beta():
    """L_step must enter the objective through `pondernet_beta`.

    Doubling beta must roughly double the decoder's gradient. This catches a subtler
    regression than the on/off test: a loss that is connected but whose weight is ignored.
    """
    # One model, two backward passes: the weights must be identical across the two runs,
    # otherwise the comparison measures initialisation noise rather than beta.
    model = _build_model(pondernet=True)
    batch = _synthetic_batch(model)
    grads = {}
    for beta in (1.0, 2.0):
        model.pondernet_beta = beta
        _backward(model, batch)
        grads[beta] = _grad_magnitude(model, GROUPS["decoder"])

    assert grads[1.0] > 1e-6 and grads[2.0] > 1e-6
    ratio = grads[2.0] / grads[1.0]
    assert 1.8 < ratio < 2.2, (
        f"decoder gradient did not scale with pondernet_beta (ratio={ratio:.3f}, "
        f"expected ~2.0) - beta is not weighting L_step correctly"
    )


def test_non_pondernet_path_still_propagates_gradient():
    """Control: the original SIM-CoT path was never broken and must stay correct."""
    model = _build_model(pondernet=False)
    _backward(model, _synthetic_batch(model))

    decoder_grad = _grad_magnitude(model, GROUPS["decoder"])
    assert decoder_grad > 1e-6, (
        f"aux decoder lost gradient on the non-PonderNet path (sum|grad|={decoder_grad})"
    )


def test_returned_loss_components_are_detached():
    """The logging contract must survive the fix.

    The returned dict is used for TensorBoard scalars; those entries must stay detached so
    they cannot hold the graph alive. The fix must restore gradient flow WITHOUT making the
    reported components graph-connected.
    """
    model = _build_model(pondernet=True)
    out = _backward(model, _synthetic_batch(model))

    for key in ("ce_loss", "distill_loss", "ref_ce_loss", "explain_loss", "kl_geom"):
        if key in out and isinstance(out[key], torch.Tensor):
            assert not out[key].requires_grad, f"returned component {key!r} is not detached"
