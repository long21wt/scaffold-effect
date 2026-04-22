import os
import argparse
import json
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

try:
    import scienceplots

    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


SCAFFOLD_LAYER = 33  # 1-indexed; the last layer BEFORE the decision forms
# (layer 34 is where signal first appears; layer 33 is the input to that step)

MDD_TOKEN = "Major"
CONTROL_TOKEN = "Control"

TABULAR_PREAMBLE = "You are given patient clinical information."
MRI_PREAMBLE = (
    "You are given patient clinical information and their MRI data "
    "(brain parcellation volume, visualization of brain regions)."
)

# Candidate phrases to test in Step 3
# Organised into semantic categories so we can interpret the results
CANDIDATE_PHRASES = {
    "MRI / neuroimaging": [
        "You are given patient clinical information and their MRI data.",
        "Brain MRI findings are available.",
        "Neuroimaging data is provided.",
        "fMRI data is included.",
        "MRI scan results are attached.",
        "Brain scans have been performed.",
    ],
    "General clinical / diagnostic": [
        "A clinical diagnosis has been established.",
        "The patient has been evaluated by a specialist.",
        "Diagnostic results are available.",
        "Medical records are provided.",
        "The patient has been assessed for a psychiatric disorder.",
        "Clinical evaluation is complete.",
    ],
    "Authoritative framing": [
        "You are an expert clinical psychiatrist.",
        "You are a specialist in mood disorders.",
        "As a medical professional, review the following.",
        "You have extensive experience in psychiatric diagnosis.",
    ],
    "Pathology / disorder priming": [
        "The patient may have a depressive disorder.",
        "Symptoms of depression have been observed.",
        "The patient presents with mood disturbances.",
        "A psychiatric condition is suspected.",
        "The patient reports persistent low mood.",
        "Mental health concerns have been flagged.",
    ],
    "Neutral / unrelated": [
        "The weather is sunny today.",
        "This is a test of the system.",
        "Please process the following information.",
        "Data is provided below.",
        "Answer the following question carefully.",
        "You are a helpful assistant.",
    ],
    "Structural / format priming": [
        "Return your answer as JSON.",
        "Respond only with a JSON object.",
        'Output: {"category":',
        "The answer is:",
        "Classification result:",
    ],
    "Negation / opposite priming": [
        "The patient is healthy and shows no symptoms.",
        "No psychiatric disorder has been detected.",
        "The patient is a control subject.",
        "All clinical indicators are within normal range.",
    ],
}


class ScaffoldModel:
    def __init__(self, model_name: str):
        print(f"\nLoading {model_name} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        self.mdd_tok_id = self.tokenizer.encode(MDD_TOKEN, add_special_tokens=False)[0]
        self.ctrl_tok_id = self.tokenizer.encode(
            CONTROL_TOKEN, add_special_tokens=False
        )[0]
        print(f"  MDD token id:  {self.mdd_tok_id}  '{MDD_TOKEN}'")
        print(f"  Ctrl token id: {self.ctrl_tok_id}  '{CONTROL_TOKEN}'")

        # LM components (confirmed paths)
        self.lm_head = self.model.lm_head
        self.n_hidden = self.lm_head.weight.shape[1]
        self.final_ln = self.model.model.language_model.norm
        print(f"  Hidden size: {self.n_hidden}")

        # Collect all layer modules for hook-free hidden state extraction
        # We use output_hidden_states=True in generate()

    def _prepare(self, messages):
        tmpl = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        img_in, vid_in = process_vision_info(messages)
        return self.processor(
            text=[tmpl],
            images=img_in,
            videos=vid_in,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

    def get_label_hidden_state(
        self,
        messages: List[Dict],
        target_layer: int = SCAFFOLD_LAYER,  # 1-indexed
        max_new_tokens: int = 80,
    ) -> Optional[Tuple[np.ndarray, str, float, float]]:
        """
        Generate and return the hidden state at `target_layer` at the
        decode step where the label token is produced.

        Returns (hidden_state_np, found_label, p_mdd_final, p_ctrl_final)
        or None if label not found.
        """
        inputs = self._prepare(messages)
        with torch.inference_mode():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        in_len = inputs["input_ids"].shape[1]
        new_ids = gen_out.sequences[0, in_len:].tolist()

        label_step = None
        found_label = None
        for i, tid in enumerate(new_ids):
            if tid == self.mdd_tok_id:
                label_step = i
                found_label = "MDD"
                break
            if tid == self.ctrl_tok_id:
                label_step = i
                found_label = "Control"
                break

        if found_label is None:
            return None

        # hidden_states[label_step]: tuple of (n_layers+1) tensors (1,1,hidden)
        hs_at_step = gen_out.hidden_states[label_step]
        # index 0 = embedding; target_layer (1-indexed) → index target_layer
        h = hs_at_step[target_layer][0, 0, :].float().cpu().numpy()  # (hidden,)

        # Also get final-layer P(MDD) and P(Control)
        h_final = (
            hs_at_step[-1][0, 0, :]
            .unsqueeze(0)
            .to(device=next(self.final_ln.parameters()).device, dtype=torch.bfloat16)
        )
        with torch.inference_mode():
            logits = self.lm_head(
                self.final_ln(h_final).to(self.lm_head.weight.device)
            ).float()[0]
            probs = torch.softmax(logits, dim=-1)
        p_mdd = probs[self.mdd_tok_id].item()
        p_ctrl = probs[self.ctrl_tok_id].item()

        return h, found_label, p_mdd, p_ctrl

    def get_p_mdd_final(self, messages, max_new_tokens=80):
        """Quick scalar: P(MDD) at final layer at label step. Returns None if label absent."""
        result = self.get_label_hidden_state(messages, max_new_tokens=max_new_tokens)
        if result is None:
            return None, None, None
        _, found_label, p_mdd, p_ctrl = result
        return p_mdd, p_ctrl, found_label


def build_messages(text, preamble: str):
    prompt = (
        preamble + "\n"
        "Classify the patient into one of the following categories:\n"
        "- Major Depressive Disorder\n- Control (no disorder detected)\n\n"
        "Return your answer as a JSON object with two fields:\n"
        '- "category": the chosen category\n'
        '- "explanation": a short reasoning\n\n'
        f"Patient data:\n{text}\n\n"
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful medical assistant in clinical psychiatry.",
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def load_patients(txt_path, label, n=None):
    out = []
    for f in sorted(os.listdir(txt_path)):
        if not f.lower().endswith(".txt"):
            continue
        with open(os.path.join(txt_path, f)) as fh:
            text = fh.read()
        out.append((os.path.splitext(f)[0], text, label))
        if n and len(out) >= n:
            break
    return out


def extract_scaffold_direction(
    model: ScaffoldModel,
    patients: List[Tuple],
    output_dir: str,
    target_layer: int = SCAFFOLD_LAYER,
) -> np.ndarray:
    """
    For each patient, extract hidden state at target_layer under:
      (a) tabular-only preamble
      (b) MRI preamble
    Scaffold direction = mean(h_mri) - mean(h_tabular).
    """
    h_tabular = []
    h_mri = []

    print(f"\nStep 1: Extracting layer-{target_layer} hidden states ...")
    print(
        f"  Patients: {len(patients)}  ×  2 conditions = {2 * len(patients)} forward passes"
    )

    skips = 0
    for pid, text, true_label in tqdm(patients, desc="Extracting"):
        for preamble, store in [
            (TABULAR_PREAMBLE, h_tabular),
            (MRI_PREAMBLE, h_mri),
        ]:
            msgs = build_messages(text, preamble)
            result = model.get_label_hidden_state(msgs, target_layer=target_layer)
            if result is None:
                tqdm.write(f"  [SKIP] {pid}/{preamble[:20]}: label not generated")
                skips += 1
                store.append(None)
            else:
                h, found_label, p_mdd, p_ctrl = result
                store.append(h)

    # Filter to matched pairs (both conditions succeeded)
    pairs = [
        (ht, hm)
        for ht, hm in zip(h_tabular, h_mri)
        if ht is not None and hm is not None
    ]
    print(f"  Valid pairs: {len(pairs)}  (skipped {skips})")

    h_tab_arr = np.stack([p[0] for p in pairs])  # (n, hidden)
    h_mri_arr = np.stack([p[1] for p in pairs])

    scaffold_dir = h_mri_arr.mean(0) - h_tab_arr.mean(0)  # (hidden,)
    scaffold_dir_norm = scaffold_dir / (np.linalg.norm(scaffold_dir) + 1e-9)

    # Save
    path = os.path.join(output_dir, "scaffold_direction.npz")
    np.savez(
        path,
        scaffold_dir=scaffold_dir,
        scaffold_dir_norm=scaffold_dir_norm,
        h_tabular=h_tab_arr,
        h_mri=h_mri_arr,
    )
    print(f"  Saved: {path}")
    print(f"  Scaffold direction norm: {np.linalg.norm(scaffold_dir):.4f}")
    print(
        f"  Mean cos(h_mri, dir): "
        f"{
            np.dot(
                h_mri_arr / (np.linalg.norm(h_mri_arr, axis=1, keepdims=True) + 1e-9),
                scaffold_dir_norm,
            ).mean():.4f}"
    )

    return scaffold_dir_norm


def single_token_search(
    model: ScaffoldModel,
    scaffold_dir: np.ndarray,
    patients: List[Tuple],
    output_dir: str,
    target_layer: int = SCAFFOLD_LAYER,
    top_k: int = 100,
    vocab_sample: int = 5000,  # test this many tokens (full vocab=151936 is too slow)
) -> List[Tuple]:
    """
    For a sample of vocabulary tokens, prepend the token to the tabular prompt
    and measure cosine similarity of the resulting layer-{target_layer} hidden
    state shift to scaffold_dir.

    We use a single patient for speed, then re-rank by effect size.
    """
    print(f"\nStep 2: Single-token vocabulary search (sample={vocab_sample}) ...")

    pid, text, true_label = patients[0]

    # Sample vocab: prioritize alphabetic single-word tokens, skip special tokens
    tokenizer = model.tokenizer
    all_ids = list(range(tokenizer.vocab_size))

    # Filter: only tokens that decode to a printable word (skip bytes, specials)
    candidate_ids = []
    for tid in all_ids:
        tok = tokenizer.decode([tid])
        if tok.strip() and tok.replace(" ", "").isalpha() and len(tok.strip()) >= 2:
            candidate_ids.append(tid)
        if len(candidate_ids) >= vocab_sample:
            break

    print(f"  Filtered vocab candidates: {len(candidate_ids)}")

    # Baseline hidden state (no extra prefix)
    msgs_base = build_messages(text, TABULAR_PREAMBLE)
    res_base = model.get_label_hidden_state(msgs_base, target_layer=target_layer)
    if res_base is None:
        print("  [ERR] Baseline failed")
        return []
    h_base = res_base[0]

    results = []
    for tid in tqdm(candidate_ids, desc="Token search"):
        tok_str = tokenizer.decode([tid])
        # Prepend token to preamble
        augmented_preamble = tok_str.strip() + ". " + TABULAR_PREAMBLE
        msgs = build_messages(text, augmented_preamble)
        res = model.get_label_hidden_state(msgs, target_layer=target_layer)
        if res is None:
            continue
        h_aug = res[0]

        shift = h_aug - h_base
        cos_sim = float(np.dot(shift / (np.linalg.norm(shift) + 1e-9), scaffold_dir))
        p_mdd = res[2]
        results.append((cos_sim, p_mdd, tid, tok_str.strip()))

    results.sort(reverse=True)
    top = results[:top_k]
    bottom = results[-top_k:]  # most anti-scaffold (interesting too)

    print(f"\n  Top-{min(20, top_k)} scaffold-aligned single tokens:")
    print(f"  {'Rank':<5} {'CosSim':>8} {'P(MDD)':>8}  Token")
    for i, (cos, pm, tid, tok) in enumerate(top[:20]):
        print(f"  {i + 1:<5} {cos:>8.4f} {pm:>8.4f}  '{tok}'")

    print("\n  Top-10 anti-scaffold tokens (push toward Control):")
    for i, (cos, pm, tid, tok) in enumerate(bottom[:10]):
        print(f"  {'─':>5} {cos:>8.4f} {pm:>8.4f}  '{tok}'")

    # Save
    path = os.path.join(output_dir, "token_search_results.json")
    with open(path, "w") as f:
        json.dump(
            {
                "top_scaffold": [(c, p, t, tok) for c, p, t, tok in top],
                "anti_scaffold": [(c, p, t, tok) for c, p, t, tok in bottom],
            },
            f,
            indent=2,
        )
    print(f"  Saved: {path}")
    return top


def phrase_search(
    model: ScaffoldModel,
    scaffold_dir: np.ndarray,
    patients: List[Tuple],
    output_dir: str,
    target_layer: int = SCAFFOLD_LAYER,
) -> List[Dict]:
    """
    Test each candidate phrase by:
      1. Measuring cosine similarity of the hidden state shift to scaffold_dir
      2. Measuring actual P(MDD) at the final layer
    Both across multiple patients for reliability.
    """
    print(
        f"\nStep 3: Phrase search ({sum(len(v) for v in CANDIDATE_PHRASES.values())} phrases × {len(patients)} patients) ..."
    )

    # Baseline: tabular-only P(MDD) per patient
    print("  Computing baselines ...")
    baselines_h = {}  # pid → h at target_layer
    baselines_pm = {}  # pid → P(MDD)

    for pid, text, true_label in tqdm(patients, desc="Baseline"):
        msgs = build_messages(text, TABULAR_PREAMBLE)
        res = model.get_label_hidden_state(msgs, target_layer=target_layer)
        if res is None:
            continue
        baselines_h[pid] = res[0]
        baselines_pm[pid] = res[2]

    print(f"  Valid baselines: {len(baselines_h)}")

    all_results = []

    for category, phrases in CANDIDATE_PHRASES.items():
        print(f"\n  Category: {category}")
        for phrase in phrases:
            cos_sims = []
            p_mdds = []
            p_shifts = []  # P(MDD) shift vs baseline

            for pid, text, true_label in patients:
                if pid not in baselines_h:
                    continue
                msgs = build_messages(text, phrase)
                res = model.get_label_hidden_state(msgs, target_layer=target_layer)
                if res is None:
                    continue

                h_aug = res[0]
                p_mdd = res[2]

                shift = h_aug - baselines_h[pid]
                cos_sim = float(
                    np.dot(shift / (np.linalg.norm(shift) + 1e-9), scaffold_dir)
                )
                cos_sims.append(cos_sim)
                p_mdds.append(p_mdd)
                p_shifts.append(p_mdd - baselines_pm[pid])

            if not cos_sims:
                continue

            result = {
                "category": category,
                "phrase": phrase,
                "cos_sim_mean": float(np.mean(cos_sims)),
                "cos_sim_std": float(np.std(cos_sims)),
                "p_mdd_mean": float(np.mean(p_mdds)),
                "p_mdd_std": float(np.std(p_mdds)),
                "p_shift_mean": float(np.mean(p_shifts)),
                "p_shift_std": float(np.std(p_shifts)),
                "n": len(cos_sims),
            }
            all_results.append(result)
            print(
                f"    cos={result['cos_sim_mean']:+.3f}  "
                f"P(MDD)={result['p_mdd_mean']:.3f}  "
                f"shift={result['p_shift_mean']:+.3f}  "
                f"'{phrase[:60]}'"
            )

    # Sort by cosine similarity
    all_results.sort(key=lambda x: x["cos_sim_mean"], reverse=True)

    print(f"\n{'─' * 80}")
    print("PHRASE SEARCH RANKING (by cosine similarity to scaffold direction)")
    print(f"{'─' * 80}")
    print(f"{'Rank':<5} {'CosSim':>8} {'P(MDD)':>8} {'Shift':>8}  Category / Phrase")
    print(f"{'─' * 80}")
    for i, r in enumerate(all_results):
        print(
            f"  {i + 1:<4} {r['cos_sim_mean']:>+8.4f} {r['p_mdd_mean']:>8.4f} "
            f"{r['p_shift_mean']:>+8.4f}  [{r['category']}] '{r['phrase'][:55]}'"
        )

    path = os.path.join(output_dir, "phrase_search_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {path}")
    return all_results


def plot_phrase_results(all_results: List[Dict], output_dir: str):
    """Bar chart of cosine similarity and P(MDD) shift by phrase category."""
    if not all_results:
        return

    from collections import defaultdict

    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r["category"]].append(r)

    categories = list(by_cat.keys())
    cat_cos = [np.mean([r["cos_sim_mean"] for r in by_cat[c]]) for c in categories]
    cat_shift = [np.mean([r["p_shift_mean"] for r in by_cat[c]]) for c in categories]

    # Sort by cosine similarity
    order = np.argsort(cat_cos)[::-1]
    categories = [categories[i] for i in order]
    cat_cos = [cat_cos[i] for i in order]
    cat_shift = [cat_shift[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(categories)))[::-1]  # pylint: disable=no-member

    for ax, vals, ylabel, title in [
        (
            axes[0],
            cat_cos,
            "Cosine sim to scaffold direction",
            "Alignment with scaffold direction\n(by phrase category)",
        ),
        (
            axes[1],
            cat_shift,
            "ΔP(MDD) vs tabular baseline",
            "P(MDD) shift vs tabular-only\n(by phrase category)",
        ),
    ]:
        bars = ax.barh(
            range(len(categories)), vals, color=colors, edgecolor="white", linewidth=0.5
        )
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=8)
        ax.set_xlabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.suptitle("Scaffold Direction Search - FOR2107", fontsize=10, y=1.02)
    fig.tight_layout()
    out = os.path.join(output_dir, "scaffold_phrase_results.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_top_phrases_scatter(all_results, output_dir):
    """Scatter: cosine sim (x) vs P(MDD) shift (y), coloured by category."""
    if not all_results:
        return

    categories = list({r["category"] for r in all_results})
    cmap = plt.cm.tab10  # pylint: disable=no-member
    cdict = {c: cmap(i / len(categories)) for i, c in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in all_results:
        ax.scatter(
            r["cos_sim_mean"],
            r["p_shift_mean"],
            color=cdict[r["category"]],
            s=60,
            alpha=0.8,
            zorder=3,
        )
        if abs(r["cos_sim_mean"]) > 0.1 or abs(r["p_shift_mean"]) > 0.05:
            ax.annotate(
                r["phrase"][:35],
                (r["cos_sim_mean"], r["p_shift_mean"]),
                fontsize=4.5,
                alpha=0.75,
                xytext=(4, 2),
                textcoords="offset points",
            )

    # Legend
    for cat, color in cdict.items():
        ax.scatter([], [], color=color, label=cat, s=40)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper left")

    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Cosine similarity to scaffold direction", fontsize=9)
    ax.set_ylabel("ΔP(MDD) vs tabular baseline", fontsize=9)
    ax.set_title(
        "Equivalence class of scaffold-activating phrases — FOR2107\n"
        "Top-right quadrant = same effect as MRI preamble",
        fontsize=9,
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "scaffold_scatter.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--txt_mdd_path", required=True)
    p.add_argument("--txt_ctrl_path", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument(
        "--n_patients",
        type=int,
        default=15,
        help="Patients per class for direction extraction",
    )
    p.add_argument("--scaffold_layer", type=int, default=SCAFFOLD_LAYER)
    p.add_argument("--output_dir", default="./scaffold_results")
    p.add_argument(
        "--load_direction",
        default=None,
        help="Path to saved scaffold_direction.npz — skip Step 1",
    )
    p.add_argument(
        "--skip_token_search",
        action="store_true",
        help="Skip Step 2 (single-token vocab search, slow)",
    )
    p.add_argument("--vocab_sample", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = ScaffoldModel(args.model_name)

    print(f"\nLoading {args.n_patients} patients per class ...")
    mdd_pts = load_patients(args.txt_mdd_path, "MDD", args.n_patients)
    ctrl_pts = load_patients(args.txt_ctrl_path, "Control", args.n_patients)
    # Use MDD patients for direction extraction (they show the clearest shift)
    # Use all patients for phrase search
    direction_patients = mdd_pts
    all_patients = mdd_pts + ctrl_pts
    print(f"  MDD={len(mdd_pts)}  Control={len(ctrl_pts)}")

    if args.load_direction:
        data = np.load(args.load_direction)
        scaffold_dir = data["scaffold_dir_norm"]
        print(f"\nLoaded scaffold direction from {args.load_direction}")
        print(f"  Norm: {np.linalg.norm(data['scaffold_dir']):.4f}")
    else:
        scaffold_dir = extract_scaffold_direction(
            model, direction_patients, args.output_dir, args.scaffold_layer
        )

    if not args.skip_token_search:
        top_tokens = single_token_search(
            model,
            scaffold_dir,
            mdd_pts[:3],  # 3 patients for speed
            args.output_dir,
            args.scaffold_layer,
            vocab_sample=args.vocab_sample,
        )
    else:
        print("\nSkipping Step 2 (single-token search)")

    phrase_results = phrase_search(
        model, scaffold_dir, mdd_pts, args.output_dir, args.scaffold_layer
    )

    print("\nGenerating figures ...")
    plot_phrase_results(phrase_results, args.output_dir)
    plot_top_phrases_scatter(phrase_results, args.output_dir)

    print(f"\nDone. Results in {args.output_dir}")
