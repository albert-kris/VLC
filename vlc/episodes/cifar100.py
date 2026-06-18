"""CIFAR-100 episode builder and criterion definitions.

CIFAR-100 has 100 fine classes grouped into 20 coarse superclasses.
We define additional grouping criteria over the 20 superclasses.
"""

from __future__ import annotations

# 20 coarse superclasses (index 0-19)
CIFAR100_COARSE = [
    "aquatic_mammals",       # 0
    "fish",                  # 1
    "flowers",               # 2
    "food_containers",       # 3
    "fruit_and_vegetables",  # 4
    "household_electrical_devices",  # 5
    "household_furniture",   # 6
    "insects",               # 7
    "large_carnivores",      # 8
    "large_man-made_outdoor_things",  # 9
    "large_natural_outdoor_scenes",   # 10
    "large_omnivores_and_herbivores", # 11
    "medium_mammals",        # 12
    "non-insect_invertebrates",       # 13
    "people",                # 14
    "reptiles",              # 15
    "small_mammals",         # 16
    "trees",                 # 17
    "vehicles_1",            # 18
    "vehicles_2",            # 19
]

CRITERIA = {
    "by_kingdom": {
        "instruction": "Group these images by biological kingdom or major category: animals, plants, humans, objects.",
        "description": "animals / plants / humans / objects",
        "coarse_to_super": {
            0: 0,   # aquatic_mammals -> animals
            1: 0,   # fish -> animals
            2: 1,   # flowers -> plants
            3: 3,   # food_containers -> objects
            4: 1,   # fruit_and_vegetables -> plants
            5: 3,   # household_electrical_devices -> objects
            6: 3,   # household_furniture -> objects
            7: 0,   # insects -> animals
            8: 0,   # large_carnivores -> animals
            9: 3,   # large_man-made_outdoor_things -> objects
            10: 1,  # large_natural_outdoor_scenes -> plants/nature
            11: 0,  # large_omnivores_and_herbivores -> animals
            12: 0,  # medium_mammals -> animals
            13: 0,  # non-insect_invertebrates -> animals
            14: 2,  # people -> humans
            15: 0,  # reptiles -> animals
            16: 0,  # small_mammals -> animals
            17: 1,  # trees -> plants
            18: 3,  # vehicles_1 -> objects
            19: 3,  # vehicles_2 -> objects
        },
        "n_super": 4,
    },
    "by_activity": {
        "instruction": "Group these images by activity domain: nature/wildlife, urban/man-made, domestic/home, people.",
        "description": "nature / urban / domestic / people",
        "coarse_to_super": {
            0: 0,   # aquatic_mammals -> nature
            1: 0,   # fish -> nature
            2: 0,   # flowers -> nature
            3: 2,   # food_containers -> domestic
            4: 0,   # fruit_and_vegetables -> nature
            5: 2,   # household_electrical_devices -> domestic
            6: 2,   # household_furniture -> domestic
            7: 0,   # insects -> nature
            8: 0,   # large_carnivores -> nature
            9: 1,   # large_man-made_outdoor_things -> urban
            10: 0,  # large_natural_outdoor_scenes -> nature
            11: 0,  # large_omnivores_and_herbivores -> nature
            12: 0,  # medium_mammals -> nature
            13: 0,  # non-insect_invertebrates -> nature
            14: 3,  # people -> people
            15: 0,  # reptiles -> nature
            16: 2,  # small_mammals -> domestic (pets)
            17: 0,  # trees -> nature
            18: 1,  # vehicles_1 -> urban
            19: 1,  # vehicles_2 -> urban
        },
        "n_super": 4,
    },
    "by_environment": {
        "instruction": "Group these images by typical environment: water, land/ground, air, indoor.",
        "description": "water / land / air / indoor",
        "coarse_to_super": {
            0: 0,   # aquatic_mammals -> water
            1: 0,   # fish -> water
            2: 1,   # flowers -> land
            3: 3,   # food_containers -> indoor
            4: 3,   # fruit_and_vegetables -> indoor
            5: 3,   # household_electrical_devices -> indoor
            6: 3,   # household_furniture -> indoor
            7: 1,   # insects -> land
            8: 1,   # large_carnivores -> land
            9: 1,   # large_man-made_outdoor_things -> land
            10: 1,  # large_natural_outdoor_scenes -> land
            11: 1,  # large_omnivores_and_herbivores -> land
            12: 1,  # medium_mammals -> land
            13: 0,  # non-insect_invertebrates -> water
            14: 3,  # people -> indoor
            15: 1,  # reptiles -> land
            16: 3,  # small_mammals -> indoor (pets)
            17: 1,  # trees -> land
            18: 2,  # vehicles_1 -> air (planes, rockets)
            19: 0,  # vehicles_2 -> water (boats) / land mix -> water
        },
        "n_super": 4,
    },
}
