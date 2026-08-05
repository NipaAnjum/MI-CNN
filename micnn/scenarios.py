"""Mapping of the dataset's 35 perturbation labels to 7 fall categories.

The public dataset annotates each perturbation event with a fine-grained label
(1-35); label 0 means no perturbation was active. For the leave-one-scenario-out
analysis these labels are grouped into seven coarse perturbation *categories*, so
that a whole family of disturbances can be held out of training at once.

Holding out a single fine label would not be a real test of generalisation --- the
other 17 roll perturbations, say, would still be in the training set. Grouping
first means each fold removes an entire kind of instability.
"""

from typing import Dict

LATERAL = "Lateral (Roll/ML)"
ANTERIOR_POSTERIOR = "Forward/Backward (AP)"
SLIP = "Slip (Pitch)"
TRIP = "Trip"
HEIGHTS = "Fall from heights (Vertigo)"
OBJECT_AVOIDANCE = "Object Avoidance"
SYNCOPE = "Syncope"

# Fine perturbation label -> coarse category. Label 0 (no perturbation) is
# deliberately absent: it denotes control/negative windows, not a scenario.
LABEL_TO_CATEGORY: Dict[int, str] = {}

# 1-18: scene roll about the anterior-posterior axis, indoors and outdoors, at
# 10/20/30 degrees clockwise and counter-clockwise. 19: mediolateral translation.
for _label in range(1, 20):
    LABEL_TO_CATEGORY[_label] = LATERAL

# 20-21: anterior-posterior scene translation along a corridor.
for _label in (20, 21):
    LABEL_TO_CATEGORY[_label] = ANTERIOR_POSTERIOR

# 22, 23, 28: scene pitch paired with a slippery-surface context (wet floor, oil).
for _label in (22, 23, 28):
    LABEL_TO_CATEGORY[_label] = SLIP

LABEL_TO_CATEGORY[29] = TRIP

# 24-27, 32-35: height exposure and vertigo-inducing scenes (roof beams,
# electricity pole, free fall, virtual stairs), with and without a self-avatar.
for _label in (24, 25, 26, 27, 32, 33, 34, 35):
    LABEL_TO_CATEGORY[_label] = HEIGHTS

LABEL_TO_CATEGORY[31] = OBJECT_AVOIDANCE
LABEL_TO_CATEGORY[30] = SYNCOPE

# Human-readable name per fine label, used for the per-sub-scenario breakdown
# that explains *which* disturbances inside a held-out category are missed.
LABEL_NAMES: Dict[int, str] = {
    0: "No perturbation (control)",
    1: "Roll Indoor1 CW10",
    2: "Roll Indoor1 CW20",
    3: "Roll Indoor1 CW30",
    4: "Roll Indoor1 CCW10",
    5: "Roll Indoor1 CCW20",
    6: "Roll Indoor1 CCW30",
    7: "Roll Indoor2 CW10",
    8: "Roll Indoor2 CW20",
    9: "Roll Indoor2 CW30",
    10: "Roll Indoor2 CCW10",
    11: "Roll Indoor2 CCW20",
    12: "Roll Indoor2 CCW30",
    13: "Roll Outdoor CW10",
    14: "Roll Outdoor CW20",
    15: "Roll Outdoor CW30",
    16: "Roll Outdoor CCW10",
    17: "Roll Outdoor CCW20",
    18: "Roll Outdoor CCW30",
    19: "ML-Axis - Kitchen",
    20: "AP - Corridor Forward",
    21: "AP - Corridor Backward",
    22: "Pitch - Bathroom (wet)",
    23: "Pitch - Fridge (wet)",
    24: "Roof Beam - Vertigo",
    25: "Roof Beam - Vertigo NoAvatar",
    26: "Simple Roof - Vertigo",
    27: "Simple Roof - Vertigo NoAvatar",
    28: "Pitch Outdoor - Car Oil",
    29: "Trip - Sidewalk",
    30: "Bedroom Syncope",
    31: "Garden - Object Avoidance",
    32: "Electricity Pole - Vertigo",
    33: "Electricity Pole - NoAvatar",
    34: "Free Fall",
    35: "Climbing Virtual Stairs",
}

CATEGORIES = sorted(set(LABEL_TO_CATEGORY.values()))


def check_labels(present_labels) -> None:
    """Raise if the data contains a perturbation label with no category mapping."""
    unmapped = [int(l) for l in present_labels if int(l) != 0 and int(l) not in LABEL_TO_CATEGORY]
    if unmapped:
        raise ValueError(f"Perturbation labels with no category mapping: {unmapped}")
