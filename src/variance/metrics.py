"""
Visual-similarity metric module for the v2 feedback experiment.

Given a (reference_image, task_image) pair, each metric returns a DISTANCE
(lower = better). Backbone metrics use cosine distance (1 - cosine similarity)
on a single pooled feature vector. Models are loaded once and cached; inference
runs on MPS if available, else CPU.

This is a deliberately raw baseline: distances are returned as-is, with NO
normalization, whitening, calibration, or corpus statistics.

Metric catalog
--------------
  FED (can be fed to the agent):     dinov3, siglip2, convnextv2, dreamsim
  HELD-OUT (offline evaluation only): inception, lpips

The active metric *set* and each metric's *checkpoint* are chosen by the caller
(env-driven in feedback_mcp_server.py). The defaults below are distilled /
CPU-friendly variants; swap to larger ones later by overriding the checkpoint
without touching this file.

Heavy deps (timm / dreamsim / lpips / transformers) are imported lazily inside
each metric, so building only `dinov3` pulls in nothing beyond torch +
transformers (the original v2 dependency set).
"""

import os

# Default checkpoints — distilled / CPU-friendly. Override per-metric via the
# caller (e.g. SIGLIP2_CHECKPOINT env var); see feedback_mcp_server.py.
DEFAULT_CHECKPOINTS = {
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",  # transformers / HF
    "siglip2": "vit_base_patch16_siglip_256.v2_webli",      # timm
    "convnextv2": "convnextv2_atto.fcmae_ft_in1k",          # timm (~3.7M params)
    "dreamsim": "ensemble",                                  # dreamsim_type
    "inception": "inception_v3.tv_in1k",                    # timm (held-out)
    "lpips": "alex",                                         # lpips net (held-out)
}

# Loader family per metric.
KIND = {
    "dinov3": "transformers",
    "siglip2": "timm",
    "convnextv2": "timm",
    "inception": "timm",
    "dreamsim": "dreamsim",
    "lpips": "lpips",
}

# Short labels for the per-metric feedback breakdown (ensemble_vector mode).
LABELS = {
    "dinov3": "DINOv3",
    "siglip2": "SigLIP2",
    "convnextv2": "ConvNeXt2",
    "dreamsim": "DreamSim",
    "inception": "Inception",
    "lpips": "LPIPS",
}


# Metrics whose MPS kernels are non-deterministic and must run on CPU for the
# scores to be reproducible. ConvNeXt-v2's GRN / LayerNorm2d path returns
# run-to-run-varying outputs on Apple MPS (observed ~0.07 swing in cosine
# distance for identical inputs); CPU is bit-exact. The atto default is tiny so
# the CPU cost is negligible. Override per metric with <METRIC>_DEVICE, or pin
# everything with METRIC_DEVICE, if you swap to a larger variant.
DEFAULT_DEVICE_OVERRIDES = {"convnextv2": "cpu"}


def resolve_device(spec=None):
    """Turn a device spec into a torch.device. None => MPS if available, else
    CPU (no CUDA on the target macOS boxes)."""
    import torch

    if spec:
        return torch.device(spec)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_device_for(name):
    """Per-metric default device: the auto choice, except for metrics pinned to
    CPU in DEFAULT_DEVICE_OVERRIDES for reproducibility."""
    return resolve_device(DEFAULT_DEVICE_OVERRIDES.get(name))


# Back-compat alias.
def get_device():
    return resolve_device()


class _Metric:
    def __init__(self, name, checkpoint, device):
        self.name = name
        self.checkpoint = checkpoint
        self.device = device

    def distance(self, ref_path: str, render_path: str) -> float:
        raise NotImplementedError


class _CosineMixin:
    """Caches pooled embeddings by absolute path so the reference image is
    embedded once and reused across every render step."""

    def __init__(self):
        self._cache = {}

    def _compute_embedding(self, image_path: str):
        raise NotImplementedError

    def _embed(self, image_path: str):
        ap = os.path.abspath(image_path)
        cached = self._cache.get(ap)
        if cached is not None:
            return cached
        emb = self._compute_embedding(image_path)
        self._cache[ap] = emb
        return emb

    def distance(self, ref_path: str, render_path: str) -> float:
        import torch.nn.functional as F

        a = self._embed(ref_path)
        b = self._embed(render_path)
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        return 1.0 - cos


class TransformersCosineMetric(_CosineMixin, _Metric):
    """HuggingFace backbone (DINOv3). Uses pooler_output — identical to the
    original v2 feedback computation so the control stays comparable."""

    def __init__(self, name, checkpoint, device):
        _Metric.__init__(self, name, checkpoint, device)
        _CosineMixin.__init__(self)
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval()

    def _compute_embedding(self, image_path: str):
        import torch
        from transformers.image_utils import load_image

        image = load_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output.squeeze(0)


class TimmCosineMetric(_CosineMixin, _Metric):
    """timm backbone (SigLIP2, ConvNeXt-v2, Inception-v3). Pooled features via
    num_classes=0, cosine distance between them."""

    def __init__(self, name, checkpoint, device):
        _Metric.__init__(self, name, checkpoint, device)
        _CosineMixin.__init__(self)
        import timm

        self.model = (
            timm.create_model(checkpoint, pretrained=True, num_classes=0)
            .to(device)
            .eval()
        )
        try:
            cfg = timm.data.resolve_model_data_config(self.model)
        except AttributeError:  # older timm
            cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

    def _compute_embedding(self, image_path: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feat = self.model(x)
        return feat.squeeze(0)


class DreamSimMetric(_Metric):
    """Official dreamsim package. Returns a perceptual distance directly (lower
    = closer), so no cosine wrapping."""

    def __init__(self, name, checkpoint, device):
        super().__init__(name, checkpoint, device)
        from dreamsim import dreamsim

        kwargs = {"pretrained": True, "device": str(device)}
        if checkpoint and checkpoint not in ("ensemble", "default"):
            kwargs["dreamsim_type"] = checkpoint
        self.model, self.preprocess = dreamsim(**kwargs)
        self._cache = {}

    def _prep(self, image_path: str):
        from PIL import Image

        ap = os.path.abspath(image_path)
        cached = self._cache.get(ap)
        if cached is not None:
            return cached
        t = self.preprocess(Image.open(image_path).convert("RGB")).to(self.device)
        self._cache[ap] = t
        return t

    def distance(self, ref_path: str, render_path: str) -> float:
        return self.model(self._prep(ref_path), self._prep(render_path)).item()


class LPIPSMetric(_Metric):
    """Official lpips package (held-out). Render is resized to the reference's
    resolution before comparison, matching scripts/lpips_eval.py."""

    def __init__(self, name, checkpoint, device):
        super().__init__(name, checkpoint, device)
        import lpips
        import torchvision.transforms as T

        net = checkpoint or "alex"
        self.loss_fn = lpips.LPIPS(net=net).to(device).eval()
        # LPIPS expects tensors in [-1, 1], shape (1, 3, H, W).
        self._to_tensor = T.Compose(
            [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        )

    def _load(self, image_path: str, size=None):
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
        return self._to_tensor(img).unsqueeze(0)

    def distance(self, ref_path: str, render_path: str) -> float:
        import torch

        ref_t = self._load(ref_path).to(self.device)
        size = (ref_t.shape[3], ref_t.shape[2])  # (W, H) for PIL
        render_t = self._load(render_path, size=size).to(self.device)
        with torch.no_grad():
            return self.loss_fn(ref_t, render_t).item()


_FACTORY = {
    "transformers": TransformersCosineMetric,
    "timm": TimmCosineMetric,
    "dreamsim": DreamSimMetric,
    "lpips": LPIPSMetric,
}


def build_metric(name: str, checkpoint=None, device=None) -> _Metric:
    if name not in KIND:
        raise ValueError(
            f"Unknown metric '{name}'. Known: {sorted(KIND)}"
        )
    if device is None:
        device = default_device_for(name)
    elif not hasattr(device, "type"):  # string spec
        device = resolve_device(device)
    ckpt = checkpoint or DEFAULT_CHECKPOINTS[name]
    return _FACTORY[KIND[name]](name, ckpt, device)


class MetricEnsemble:
    """Holds a set of fed + held-out metrics. Loads each model once at
    construction (fail-fast on missing deps), then computes raw distances for a
    (reference, render) pair on demand.
    """

    def __init__(
        self, fed_names, heldout_names, checkpoints=None, devices=None, device=None
    ):
        # `device` is an optional global override applied to every metric;
        # `devices` overrides individual metrics by name. When neither is given
        # for a metric, its per-metric default applies (auto, or CPU for metrics
        # pinned in DEFAULT_DEVICE_OVERRIDES).
        checkpoints = checkpoints or {}
        devices = devices or {}
        self.device = resolve_device(device)  # representative, for logging
        self.fed_names = list(fed_names)
        self.heldout_names = list(heldout_names)
        self.metrics = {}
        for name in self.fed_names + self.heldout_names:
            if name in self.metrics:
                continue
            metric_device = devices.get(name) or device  # None => per-metric default
            self.metrics[name] = build_metric(
                name, checkpoints.get(name), metric_device
            )

    def warm(self, ref_path: str) -> None:
        """Pre-embed the reference image for cosine metrics so the first
        score_render call isn't slowed by reference embedding."""
        for m in self.metrics.values():
            embed = getattr(m, "_embed", None)
            if callable(embed):
                try:
                    embed(ref_path)
                except Exception:
                    pass

    def compute_all(self, ref_path: str, render_path: str) -> dict:
        """Raw distance for every metric (fed + held-out), insertion-ordered."""
        return {
            name: m.distance(ref_path, render_path)
            for name, m in self.metrics.items()
        }

    def fed_values(self, results: dict) -> dict:
        return {name: results[name] for name in self.fed_names}
