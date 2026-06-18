"""Instruction paraphrase augmentation for training robustness.

Provides a bank of paraphrases for each criterion in each dataset.
During training, each episode randomly samples one paraphrase as its criterion text.
This teaches the model to respond to varied wording, not just template phrases.
"""

from __future__ import annotations

import random


# Per-criterion paraphrase banks
PARAPHRASES: dict[str, list[str]] = {
    # 3DShapes — object_hue
    "cluster these images by object color": [
        "cluster these images by object color",
        "group the images according to the color of the main object",
        "sort these images by the hue of the central object",
        "partition the images based on what color the object is",
        "arrange these images into groups sharing the same object color",
        "categorize the images by the dominant color of the 3D shape",
        "separate these images by object hue",
    ],
    # 3DShapes — shape
    "cluster these images by object shape": [
        "cluster these images by object shape",
        "group the images by the shape of the main object",
        "sort these images according to what geometric form the object has",
        "partition the images by the 3D shape type shown",
        "arrange these images into groups sharing the same object geometry",
        "categorize the images by whether the object is a sphere, cube, cylinder, or capsule",
        "separate these images by object form",
    ],
    # 3DShapes — object_size
    "cluster these images by object size": [
        "cluster these images by object size",
        "group the images by how large or small the main object appears",
        "sort these images according to the size of the central object",
        "partition the images based on object scale",
        "arrange these images by the relative size of the 3D shape",
        "categorize the images by object size from small to large",
        "separate these images by how big the object is",
    ],
    # 3DShapes — orientation
    "cluster these images by object orientation": [
        "cluster these images by object orientation",
        "group the images by how the object is rotated or oriented",
        "sort these images according to the viewing angle of the main object",
        "partition the images based on the orientation of the 3D shape",
        "arrange these images by the azimuthal rotation of the object",
        "categorize the images by the direction the object is facing",
        "separate these images by object rotation angle",
    ],
    # CLEVR — color
    "cluster these images by the dominant object color": [
        "cluster these images by the dominant object color",
        "group the images by what color the main object is",
        "sort these images according to object color",
        "partition the images based on the dominant hue of the central object",
        "arrange these images by object color",
    ],
    # CLEVR — shape
    "cluster these images by object shape": [
        "cluster these images by object shape",
        "group the images by the geometric form of the object",
        "sort these images by whether the object is a cube, sphere, or cylinder",
        "partition the images based on the 3D shape shown",
    ],
    # CLEVR — material
    "cluster these images by surface material": [
        "cluster these images by surface material",
        "group the images by whether the object surface is rubber or metal",
        "sort these images according to the material of the main object",
        "partition the images based on surface finish (matte vs shiny)",
        "arrange these images by material type",
    ],
}


def get_paraphrases(criterion: str) -> list[str]:
    """Return paraphrase list for a criterion, or [criterion] if not in bank."""
    return PARAPHRASES.get(criterion, [criterion])


def sample_criterion(criterion: str, rng: random.Random | None = None) -> str:
    """Sample a random paraphrase of the given criterion."""
    options = get_paraphrases(criterion)
    if rng is not None:
        return rng.choice(options)
    return random.choice(options)
