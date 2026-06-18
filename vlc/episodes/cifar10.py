"""CIFAR-10 episode builder and criterion definitions."""

from __future__ import annotations

# CIFAR-10 class index -> name
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Criterion: group by animacy (living vs non-living)
# Criterion: group by size (large vs small in real world)
# Criterion: group by domestic (domestic vs wild/machine)
CRITERIA = {
    "by_animacy": {
        "instruction": "Group these images by whether the subject is a living organism or a non-living object.",
        "description": "living vs non-living",
        "class_to_super": {
            # living: bird, cat, deer, dog, frog, horse
            "bird": 0, "cat": 0, "deer": 0, "dog": 0, "frog": 0, "horse": 0,
            # non-living: airplane, automobile, ship, truck
            "airplane": 1, "automobile": 1, "ship": 1, "truck": 1,
        },
    },
    "by_size": {
        "instruction": "Group these images by the typical real-world size of the subject: large or small.",
        "description": "large vs small",
        "class_to_super": {
            # large: airplane, automobile, deer, horse, ship, truck
            "airplane": 0, "automobile": 0, "deer": 0, "horse": 0, "ship": 0, "truck": 0,
            # small: bird, cat, dog, frog
            "bird": 1, "cat": 1, "dog": 1, "frog": 1,
        },
    },
    "by_domestic": {
        "instruction": "Group these images by whether the subject is typically found in domestic/urban settings or in the wild/nature.",
        "description": "domestic/urban vs wild/nature",
        "class_to_super": {
            # domestic/urban: automobile, cat, dog, truck, ship, airplane
            "automobile": 0, "cat": 0, "dog": 0, "truck": 0, "ship": 0, "airplane": 0,
            # wild/nature: bird, deer, frog, horse
            "bird": 1, "deer": 1, "frog": 1, "horse": 1,
        },
    },
}

# Mapping: cifar10 label index -> super-class for "by_animacy" (default)
CIFAR_TO_SUPER = {
    0: 1,  # airplane -> non-living
    1: 1,  # automobile -> non-living
    2: 0,  # bird -> living
    3: 0,  # cat -> living
    4: 0,  # deer -> living
    5: 0,  # dog -> living
    6: 0,  # frog -> living
    7: 0,  # horse -> living
    8: 1,  # ship -> non-living
    9: 1,  # truck -> non-living
}
