"""Per-model adapters.

Each adapter knows how to (1) prepare any inputs a model needs, (2) build the
exact inference command + working directory + environment, and (3) locate the
video the model produced. argv is always a list (shell=False) so the Thai-named
audio file and any spaces in paths are passed safely.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import yaml
from utils import audio_frames

MODEL_ORDER = ["SadTalker", "Ditto", "IMTalker", "EchoMimic"]


class OutputNotFound(Exception):
    """Raised when a model exits cleanly but no output video can be found."""


class BaseAdapter:
    name = "Base"
    out_name = "Base.mp4"
    error_markers: tuple = ()  # log substrings that mean a silent failure

    def __init__(self, model_cfg, shared, model_dir):
        self.cfg = model_cfg
        self.shared = shared
        self.model_dir = Path(model_dir)
        self.repo_dir = Path(model_cfg["repo_dir"])
        self.python = model_cfg["python"]
        self.script = model_cfg["script"]
        self.params = model_cfg.get("params", {}) or {}

    # -- overridable hooks ---------------------------------------------------
    def prepare(self):
        """Optional pre-run step (e.g. generate a config file)."""

    def build_command(self):
        """Return (argv_list, cwd, env_dict)."""
        raise NotImplementedError

    def locate_output(self, start_ts) -> Path:
        """Return the path of the produced video, or raise OutputNotFound."""
        raise NotImplementedError

    # -- helpers -------------------------------------------------------------
    def _env(self):
        env = os.environ.copy()
        for key, val in (self.cfg.get("extra_env") or {}).items():
            env[str(key)] = str(val)
        return env

    def _newest(self, root: Path, pattern: str, start_ts: float, recursive=False):
        globber = root.rglob if recursive else root.glob
        cands = [
            p
            for p in globber(pattern)
            if p.is_file() and p.stat().st_mtime >= start_ts - 5
        ]
        if not cands:
            raise OutputNotFound(f"no file matching '{pattern}' under {root}")
        return max(cands, key=lambda p: p.stat().st_mtime)


class SadTalkerAdapter(BaseAdapter):
    name = "SadTalker"
    out_name = "SadTalker.mp4"

    def build_command(self):
        p = self.params
        self.result_dir = self.model_dir / "results"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        # --size only accepts 256 or 512; snap the shared size onto that grid.
        size = 512 if int(self.shared["size"]) >= 384 else 256
        cmd = [
            self.python,
            self.script,
            "--driven_audio",
            self.shared["driven_audio"],
            "--source_image",
            self.shared["source_image"],
            "--result_dir",
            str(self.result_dir),
            "--checkpoint_dir",
            str(p.get("checkpoint_dir", "./checkpoints")),
            "--preprocess",
            str(p.get("preprocess", "full")),
            "--size",
            str(size),
            "--batch_size",
            str(p.get("batch_size", 2)),
            "--pose_style",
            str(p.get("pose_style", 0)),
            "--expression_scale",
            str(p.get("expression_scale", 1.0)),
        ]
        if p.get("enhancer"):
            cmd += ["--enhancer", str(p["enhancer"])]
        if p.get("still"):
            cmd += ["--still"]
        return cmd, self.repo_dir, self._env()

    def locate_output(self, start_ts):
        # inference.py shutil.move()s the run dir to "<result_dir>/<stamp>.mp4".
        return self._newest(self.result_dir, "*.mp4", start_ts)


class DittoAdapter(BaseAdapter):
    name = "Ditto"
    out_name = "Ditto.mp4"

    def build_command(self):
        p = self.params
        self.output_path = self.model_dir / self.out_name
        cmd = [
            self.python,
            self.script,
            "--data_root",
            str(p["data_root"]),
            "--cfg_pkl",
            str(p["cfg_pkl"]),
            "--audio_path",
            self.shared["driven_audio"],
            "--source_path",
            self.shared["source_image"],
            "--output_path",
            str(self.output_path),
        ]
        return cmd, self.repo_dir, self._env()

    def locate_output(self, start_ts):
        if not self.output_path.exists():
            raise OutputNotFound(f"Ditto output not created: {self.output_path}")
        return self.output_path


class IMTalkerAdapter(BaseAdapter):
    name = "IMTalker"
    out_name = "IMTalker.mp4"
    # generate.py catches its own errors, prints these, and still exits 0.
    error_markers = ("Error processing", "No face detected")

    def prepare(self):
        # IMTalker's save_video() builds its ffmpeg mux command as a shell
        # string with the output path UNQUOTED, and that path is derived from
        # the source-image filename. Any space in the image name shell-splits
        # the ffmpeg output arg -> no video, but generate.py still exits 0.
        # Copy the image to a space-free name so the derived path is safe.
        # The IMTalker repo is left untouched.
        src = Path(self.shared["source_image"])
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", src.name)
        self.ref_path = self.model_dir / safe
        shutil.copy2(src, self.ref_path)

    def build_command(self):
        p = self.params
        self.res_dir = self.model_dir
        cmd = [
            self.python,
            self.script,
            "--ref_path",
            str(self.ref_path),
            "--aud_path",
            self.shared["driven_audio"],
            "--res_dir",
            str(self.res_dir),
            "--generator_path",
            str(p.get("generator_path", "./checkpoints/generator.ckpt")),
            "--renderer_path",
            str(p.get("renderer_path", "./checkpoints/renderer.ckpt")),
            "--a_cfg_scale",
            str(p.get("a_cfg_scale", 2)),
            "--seed",
            str(self.shared["seed"]),
            "--fps",
            str(self.shared["fps"]),
            "--input_size",
            str(p.get("input_size", 256)),
        ]
        if p.get("crop"):
            cmd += ["--crop"]
        return cmd, self.repo_dir, self._env()

    def locate_output(self, start_ts):
        # generate.py names the video after the reference image's stem -
        # here the space-free copy made in prepare().
        stem = self.ref_path.stem
        expected = self.res_dir / f"{stem}.mp4"
        if expected.exists() and expected.stat().st_mtime >= start_ts - 5:
            return expected
        return self._newest(self.res_dir, "*.mp4", start_ts)


class EchoMimicAdapter(BaseAdapter):
    name = "EchoMimic"
    out_name = "EchoMimic.mp4"

    def prepare(self):
        # The source image + audio come from the YAML, not from CLI flags.
        # Clone the base config and swap ONLY `test_cases`; keep every model
        # path and weight_dtype untouched. The base config is never modified.
        base = self.repo_dir / self.params.get(
            "base_config", "./configs/prompts/animation_my.yaml"
        )
        with open(base, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        img = str(Path(self.shared["source_image"]).resolve())
        aud = str(Path(self.shared["driven_audio"]).resolve())
        data["test_cases"] = {img: [aud]}
        self.gen_config = self.model_dir / "animation_generated.yaml"
        with open(self.gen_config, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)

    def build_command(self):
        p = self.params
        self.output_dir = self.model_dir / "output"
        size = int(self.shared["size"])
        # L=auto -> render exactly as many frames as the audio needs (+margin)
        # instead of a fixed 1200; rendering blank tail frames is pure waste.
        L = p.get("L", 1200)
        if str(L).lower() == "auto":
            L = audio_frames(self.shared["driven_audio"], self.shared["fps"])
        cmd = [
            self.python,
            self.script,
            "--config",
            str(self.gen_config),
            "-W",
            str(size),
            "-H",
            str(size),
            "-L",
            str(L),
            "--seed",
            str(self.shared["seed"]),
            "--cfg",
            str(p.get("cfg", 2.5)),
            "--steps",
            str(p.get("steps", 30)),
            "--fps",
            str(self.shared["fps"]),
            "--sample_rate",
            str(self.shared.get("sample_rate", 16000)),
            "--context_frames",
            str(p.get("context_frames", 12)),
            "--context_overlap",
            str(p.get("context_overlap", 3)),
            "--device",
            str(p.get("device", "cuda")),
            "--output_dir",
            str(self.output_dir),
        ]
        return cmd, self.repo_dir, self._env()

    def locate_output(self, start_ts):
        # Final muxed video ends in "_withaudio.mp4"; nested under date/time dirs.
        return self._newest(
            self.output_dir, "*_withaudio.mp4", start_ts, recursive=True
        )


ADAPTERS = {
    "SadTalker": SadTalkerAdapter,
    "Ditto": DittoAdapter,
    "IMTalker": IMTalkerAdapter,
    "EchoMimic": EchoMimicAdapter,
}
