import json
import logging
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from py_oneliner import one_liner
from matplotlib.gridspec import GridSpec

from building_footprint_segmentation.utils.py_network import (
    adjust_model,
    gpu_variable,
    convert_tensor_to_numpy,
)

logger = logging.getLogger("segmentation")

@dataclass
class ExperimentResult:
    path: str
    metrics: dict
    num_samples: int


@dataclass
class EvaluationReport:
    experiments: list = field(default_factory=list)

    def aggregate(self) -> dict:
        if not self.experiments:
            return {}
        keys = self.experiments[0].metrics.keys()
        agg = {}
        for k in keys:
            vals = np.array([e.metrics[k] for e in self.experiments])
            agg[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        return agg

@dataclass
class PlotSelection:
    n: int = 5
    mode: Literal["alpha", "random"] = "alpha"
    seed: Optional[int] = 42
    seed_per_experiment: bool = False

def _seed_for(exp_name: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{exp_name}".encode()).hexdigest()
    return int(h[:8], 16)
    
def adjust_keys(dictionary, prev_prefix, new_prefix = ""):
    return {
        (k.replace(prev_prefix, new_prefix, 1) if k.startswith(prev_prefix) else k): v
        for k, v in dictionary.items()
    }

class Evaluator:
    """
    Evaluates 'state/best.pt' checkpoints from multiple experiments.

    Parameters
    ----------
    base_dir : str
        Directory containing experiment subfolders named by timestamp.
    model_factory : Callable[[], torch.nn.Module]
        Constructs the empty model (architecture) to be loaded.
    test_loader : torch.utils.data.DataLoader
        Loader in 'test' mode (BinaryLoader with mode='test').
    metrics : Object with get_metrics/compute_mean/reset methods (same interface as the Trainer).
    device : str or torch.device
        Default: 'cuda' if available.
    threshold : float
        Binarization threshold for TP/FP/TN/FN and binary metrics.
    """

    def __init__(
        self,
        base_dir: str,
        model_factory: Callable[[], torch.nn.Module],
        test_loader,
        metrics,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.base_dir = base_dir
        self.model_factory = model_factory
        self.test_loader = test_loader
        self.metrics = metrics
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold

    def discover_experiments(self) -> list:
        """Returns a list of (name, path_best_pt) for valid subdirectories."""
        found = []
        if not os.path.isdir(self.base_dir):
            raise NotADirectoryError(self.base_dir)

        for name in sorted(os.listdir(self.base_dir)):
            exp_dir = os.path.join(self.base_dir, name)
            if not os.path.isdir(exp_dir):
                continue
            best = os.path.join(exp_dir, "state", "best.pt")
            if not os.path.isfile(best):
                logger.debug(f"Ignoring {name}: without state/best.pt")
                continue
            found.append((name, best))
        logger.info(f"found {len(found)} valid experiments")
        return found

    def _enumerate_filenames(self) -> list[str]:
        ds = self.test_loader.dataset
        # made to our BinaryLoader
        if hasattr(ds, "images") and len(ds.images) > 0:
            return [os.path.basename(str(p)) for p in ds.images]
        # fallback: iterate on loader
        names = []
        for batch in self.test_loader:
            names.extend(batch["file_name"])
        return names

    def _select_filenames(self, all_names: list[str], selection: PlotSelection, 
                          exp_name: str) -> list[str]:
        if selection.n <= 0 or not all_names:
            return []
        if selection.mode == "alpha":
            return sorted(all_names)[: selection.n]
        if selection.mode == "random":
            if selection.seed_per_experiment:
                seed = _seed_for(exp_name, selection.seed if selection.seed is not None else 0)
            else:
                if selection.seed is None:
                    raise ValueError(
                        "mode='random' without seed_per_experiment requires seed != None"
                    )
                seed = selection.seed
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(all_names), size=min(selection.n, len(all_names)), replace=False)
            return [all_names[i] for i in idx]

        raise ValueError(f"invalid mode: {selection.mode}")
    
    def _load_model(self, best_path: str) -> torch.nn.Module:
        state = torch.load(best_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model_state = state["model"]
            model_state = adjust_keys(model_state, "module.")
        else:
            # fallback: for checkpoint as single state_dict (eg.: chk_pth.pt)
            model_state = state
            
        model = self.model_factory()
        model.load_state_dict(model_state)
        #model = adjust_model(model)
        model.to(self.device)
        model.eval()
        return model

    
    @torch.no_grad()
    def evaluate_experiment(
        self,
        exp_name: str,
        best_path: str,
        selection: PlotSelection,
        all_filenames: list[str],
        save_plots: bool = False,
        show_plots: bool = False,
        exp_idx: str = "",
    ) -> ExperimentResult:
        model = self._load_model(best_path)

        if hasattr(self.metrics, "reset"):
            self.metrics.reset()

        targets_to_plot = set(
            self._select_filenames(all_filenames, selection, exp_name)
        )

        tp = fp = tn = fn = 0
        samples = 0

        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(best_path)),
            "test",
        )

        if save_plots and targets_to_plot:
            os.makedirs(out_dir, exist_ok=True)

        for batch in self.test_loader:
            images = gpu_variable(batch["images"]).to(self.device)
            gt = gpu_variable(batch["ground_truth"]).to(self.device)

            logits = model(images)
            probs = logits.sigmoid()
            preds = (probs >= self.threshold).float()

            self.metrics.get_metrics(
                ground_truth=gt,
                prediction=logits,
            )

            gt_bin = (gt >= 0.5).float()

            tp += int(((preds == 1) & (gt_bin == 1)).sum().item())
            fp += int(((preds == 1) & (gt_bin == 0)).sum().item())
            tn += int(((preds == 0) & (gt_bin == 0)).sum().item())
            fn += int(((preds == 0) & (gt_bin == 1)).sum().item())

            samples += images.shape[0]

        metrics = (
            dict(self.metrics.compute_mean())
            if hasattr(self.metrics, "compute_mean")
            else {}
        )
        
        if exp_idx:
            self._print_experiment_result(exp_name, 
                                          ExperimentResult(best_path,
                                                           metrics,
                                                           samples), 
                                          exp_idx)

        if targets_to_plot and (save_plots or show_plots):
            for batch in self.test_loader:
                raw = batch["raw_images"]
                names = batch["file_name"]

                keep = [
                    i for i, name in enumerate(names)
                    if name in targets_to_plot
                ]

                if not keep:
                    continue
                    
                images = gpu_variable(batch["images"]).to(self.device)
                gt = gpu_variable(batch["ground_truth"]).to(self.device)

                logits = model(images)
                probs = logits.sigmoid()
                preds = (probs >= self.threshold).float()
                gt_bin = (gt >= 0.5).float()
    
                self._visualize_batch(
                    raw[keep],
                    gt_bin[keep],
                    preds[keep],
                    probs[keep],
                    [names[i] for i in keep],
                    out_dir=out_dir if save_plots else None,
                    show=show_plots,
                )

        return ExperimentResult(
            path=best_path,
            metrics=metrics,
            num_samples=samples,
        )

    @staticmethod
    def _pixel_metrics(tp, fp, tn, fn) -> dict:
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": f1,
            "pixel_iou": iou,
            "pixel_accuracy": acc,
        }

    def _visualize_batch(self, raw, gt, pred, prob, names, out_dir=None, show=False):
        """
        For each sample in the batch, generate a 2x3 figure:
        raw | gt | diff | pred | overlay | (empty/summary)
        """
        raw_np = raw.cpu().numpy()
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        prob_np = prob.cpu().numpy()

        for i in range(raw_np.shape[0]):
            r = self._to_display(raw_np[i])         # (H,W,3) em [0,1] ou (H,W)
            g = gt_np[i].squeeze()
            p = pred_np[i].squeeze()
            n = prob_np[i].squeeze()

            raster = self._confusion_raster(g, p)   # (H,W,3) uint8

            fig = plt.figure(figsize=(42, 6))
            gs = GridSpec(1, 7, figure=fig)

            ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(r); ax0.set_title("Input", fontsize=26); ax0.axis("off")
            ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(g, cmap="gray", vmin=0, vmax=1); ax1.set_title("Ground truth", fontsize=26); ax1.axis("off")
            ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(raster); ax2.set_title("Difference", fontsize=26); ax2.axis("off")

            ax3 = fig.add_subplot(gs[0, 3]); ax3.imshow(p, cmap="gray", vmin=0, vmax=1); ax3.set_title("Prediction", fontsize=26); ax3.axis("off")

            overlay = self._overlay(r, raster)
            ax4 = fig.add_subplot(gs[0, 4]); ax4.imshow(overlay); ax4.set_title("Overlay", fontsize=26); ax4.axis("off")

            ax5 = fig.add_subplot(gs[0, 5]); ax5.imshow(n, cmap="hot", vmin=0, vmax=1); ax5.set_title("Output", fontsize=26); ax5.axis("off")

            ax6 = fig.add_subplot(gs[0, 6]); ax6.axis("off")
            counts = self._counts(g, p)
            total = sum(counts.values())
            ax6.text(0.05, 0.5,
                     f"{names[i]}\n"+
                     "\n".join(f"{k}: {v*100/total:.2f}%" for k, v in sorted(counts.items(), reverse=True)),
                     family="monospace", fontsize=24, va="center")

            #fig.suptitle(names[i], fontsize=12)
            fig.tight_layout()

            if out_dir:
                stem = os.path.splitext(names[i])[0]
                fig.savefig(os.path.join(out_dir, f"{stem}_eval.png"),
                            dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    @staticmethod
    def _to_display(img: np.ndarray) -> np.ndarray:
        """Aceita (C,H,W) ou (H,W); devolve (H,W,3) em [0,1]."""
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.moveaxis(img, 0, -1)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32)
        # desnormaliza heurística: se valores fora de [0,1], assume [0,255]
        if img.max() > 1.5:
            img = img / 255.0
        return np.clip(img, 0, 1)

    @staticmethod
    def _confusion_raster(g: np.ndarray, p: np.ndarray) -> np.ndarray:
        h, w = g.shape
        raster = np.zeros((h, w, 3), dtype=np.uint8)
        tp = (g == 1) & (p == 1)
        fp = (g == 0) & (p == 1)
        fn = (g == 1) & (p == 0)
        tn = (g == 0) & (p == 0)
        raster[tp] = (0, 200, 0)
        raster[fp] = (220, 0, 0)
        raster[fn] = (0, 80, 220)
        raster[tn] = (0, 0, 0)
        return raster

    @staticmethod
    def _overlay(raw: np.ndarray, raster: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        raw_u8 = (raw * 255).astype(np.uint8)
        if raw_u8.shape[:2] != raster.shape[:2]:
            raw_u8 = cv2.resize(raw_u8, (raster.shape[1], raster.shape[0]))
        mask = raster.any(axis=-1)
        out = raw_u8.copy()
        out[mask] = (alpha * raster[mask] + (1 - alpha) * raw_u8[mask]).astype(np.uint8)
        return out

    @staticmethod
    def _counts(g: np.ndarray, p: np.ndarray) -> dict:
        g = g.astype(bool); p = p.astype(bool)
        return {
            "TP (green)": int((g & p).sum()),
            "FP (red)": int((~g & p).sum()),
            "TN (black)": int((~g & ~p).sum()),
            "FN (blue)": int((g & ~p).sum()),
        }

    def run(
        self,
        selection: PlotSelection,
        save_plots: bool = True,
        show_plots: bool = False,
        output_json: Optional[str] = None,
        verbose: bool = True,
    ) -> EvaluationReport:
        report = EvaluationReport()

        all_filenames = self._enumerate_filenames()
        if selection.n > 0 and not all_filenames:
            logger.warning("test_loader without file_name; disabling visualizations")
            
        experiments = self.discover_experiments()
        total = len(experiments)

        for i, (name, best) in enumerate(experiments, start=1):
            try:
                res = self.evaluate_experiment(
                    name, best,
                    selection=selection,
                    all_filenames=all_filenames,
                    save_plots=save_plots,
                    show_plots=show_plots,
                    exp_idx=f"{i}/{total}",
                )
                report.experiments.append(res)
            except Exception as ex:
                logger.exception(f"Fail on evaluate {name}: {ex}")

        agg = report.aggregate()
        self._print_report(agg)

        if output_json:
            with open(output_json, "w") as f:
                json.dump(
                    {
                        "experiments": [
                            {"path": e.path, "num_samples": e.num_samples, "metrics": e.metrics}
                            for e in report.experiments
                        ],
                        "aggregate": agg,
                    },
                    f, indent=2,
                )
        return report
    
    @staticmethod
    def _print_report(agg: dict, use_one_liner: bool = True):
        msg = "Aggregated Evaluation (mean ± standard deviation across experiments on test dataset)"
        if use_one_liner:
            one_liner.one_line(
                tag=msg,
                tag_color="cyan",
                to_reset_data=True,
                to_new_line_data=True,
            )
            if not agg:
                one_liner.one_line(
                    tag="No experiment evaluated.",
                    tag_color="cyan",
                    to_reset_data=True,
                    to_new_line_data=True,
                )
                return
            for k, v in agg.items():
                one_liner.one_line(
                    tag=f"{k:24s}",
                    tag_data=f"{v['mean']:.4f} ± {v['std']:.4f}   (n={v['n']})",
                    tag_color="cyan",
                    to_reset_data=True,
                    to_new_line_data=True,
                )
        else:
            logger.info(msg)
            if not agg:
                logger.info("No experiment evaluated.")
                return
            for k, v in agg.items():
                logger.info(f"  {k:24s}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")

    @staticmethod
    def _print_experiment_result(
        exp_name: str,
        result,
        exp_index: str,
        use_one_liner: bool = True,
    ) -> None:
        metrics = result.metrics or {}
        header = f"[{exp_index}] {exp_name} (n={result.num_samples}):"
        parts = []
        for key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
        metric_line = "  ".join(parts) if parts else "(without metrics)"

        if use_one_liner:
            one_liner.one_line(
                tag=header,
                tag_data=metric_line,
                tag_color="cyan",
                to_reset_data=True,
            )
        else:
            #print("\n" + "=" * 79)
            logger.info(f"{header}\n    {metric_line}")
            #print("=" * 79)
