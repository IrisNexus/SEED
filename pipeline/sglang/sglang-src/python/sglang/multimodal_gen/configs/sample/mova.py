# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class MOVASamplingParams(SamplingParams):
    # Video parameters (MOVA defaults)
    height: int = 352
    width: int = 640
    num_frames: int = 193
    fps: int = 24

    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    sigma_shift: float = 5.0
    visual_shift: float = 5.0
    audio_shift: float = 5.0

    adjust_frames: bool = False

    negative_prompt: str = (
        "[redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted]"
        "[redacted][redacted][redacted][redacted][redacted][redacted]JPEG[redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted]"
        "[redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted]"
        "[redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted][redacted]"
    )


@dataclass
class MOVA_360P_SamplingParams(MOVASamplingParams):
    # Video parameters (MOVA 360P)
    height: int = 352
    width: int = 640

    # MOVA 360P supported resolutions
    supported_resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (352, 640),
            (640, 352),
        ]
    )


@dataclass
class MOVA_720P_SamplingParams(MOVASamplingParams):
    # Video parameters (MOVA 720P)
    height: int = 720
    width: int = 1280

    # MOVA 720P supported resolutions
    supported_resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (720, 1280),
            (1280, 720),
        ]
    )
