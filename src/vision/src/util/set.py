"""
Algorithm for finding a set among visible cards.
"""

import numpy as np

from .labels import deserialize_label, label_from_string, label_to_string


def find_set(card_labels):
    """
    Given a list of string labels (in the form "{shape}-{color}-{count}-{shade}"),
    return the indices of a set.
    Returns None if no set can be found.

    Uses a brute force search over all possible pairs, and checks if the (forced)
    third card is present.
    """

    label_set = set(card_labels)

    for i in range(len(card_labels)):
        card_i = card_labels[i]
        for j in range(i + 1, len(card_labels)):
            card_j = card_labels[j]

            card_i_values = np.array(deserialize_label(label_from_string(card_i)))
            card_j_values = np.array(deserialize_label(label_from_string(card_j)))

            third_card_values = np.mod(-card_i_values - card_j_values, 3)
            third_card = label_to_string(*tuple(third_card_values))

            if third_card in label_set:
                # found set, return sorted list of indices
                k = card_labels.index(third_card)
                return sorted([i, j, k])

    # can't find any sets
    return None
