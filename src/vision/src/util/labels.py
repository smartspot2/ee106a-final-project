"""
Constants and utility functions for card labels.
"""


SHAPES = ["oval", "squiggle", "diamond"]
COLORS = ["red", "purple", "green"]
COUNTS = ["1", "2", "3"]
SHADES = ["solid", "stripe", "outline"]


def label_from_string(label: str) -> int:
    """
    Serialize a card label from a string.

    Label must be of the form:
        "{shape}-{color}-{count}-{shade}"
    If duplicates, may have suffix "_{nonce}"

    Serialization is a number in base 3 with the digits
        (((shape) * 3 + color) * 3 + count) * 3 + shade
    """
    label = label.split("_")[0]
    shape, color, count, shade = label.split("-")
    shape_int = SHAPES.index(shape)
    color_int = COLORS.index(color)
    count_int = COUNTS.index(count)
    shade_int = SHADES.index(shade)

    return shape_int * 27 + color_int * 9 + count_int * 3 + shade_int


def deserialize_label(label: int) -> tuple[int, int, int, int]:
    """
    Deserialize a label from an int.

    label % 3 => shade
    (label // 3) % 3 => count
    (label // 9) % 3 => color
    (label // 27) % 3 => shape
    """

    shade = label % 3
    count = (label // 3) % 3
    color = (label // 9) % 3
    shape = (label // 27) % 3

    return shape, color, count, shade


def label_to_string(shape: int, color: int, count: int, shade: int) -> str:
    shape_str = SHAPES[shape]
    color_str = COLORS[color]
    count_str = COUNTS[count]
    shade_str = SHADES[shade]

    return f"{shape_str}-{color_str}-{count_str}-{shade_str}"
