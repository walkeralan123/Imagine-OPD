#!/usr/bin/env python3
"""
Lightweight local dataset adapters for run_vlmevalkit_local.py.

These classes are registered into VLMEvalKit's dataset namespace so the
auto-generated config can reference project-local JSON/image datasets without
patching the vendored VLMEvalKit source tree.
"""

from __future__ import annotations

from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.dataset.image_vqa import ImageVQADataset


class LocalImageMCQDataset(ImageMCQDataset):
    TYPE = "MCQ"


class LocalImageVQADataset(ImageVQADataset):
    TYPE = "VQA"


class LocalOCRBenchDataset(ImageVQADataset):
    TYPE = "VQA"


class LocalCountQADataset(ImageVQADataset):
    TYPE = "VQA"


class LocalBabyVisionDataset(ImageVQADataset):
    TYPE = "VQA"


class LocalTreeBenchDataset(ImageMCQDataset):
    TYPE = "MCQ"
