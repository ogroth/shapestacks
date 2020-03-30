"""Contains constants in the ShapeStacks rendering process."""

# colors used for the objects
OBJ_COLORS_RGBA = [
    [1, 0, 0, 1],  # red
    [0, 1, 0, 1],  # green
    [0, 0, 1, 1],  # blue
    [1, 1, 0, 1],  # yellow
    [0, 1, 1, 1],  # cyan
    [1, 0, 1, 1],  # magenta
]

# color codes for the violation segmentation maps; list index corresponds to class label
VSEG_COLOR_CODES = [
    [0, 0, 0, 1],  # black : 0 = background pixel
    [0, 1, 0, 1],  # green: 1 = lower part of the stack (stable)
    [1, 0, 0, 1],  # red: 2 = object violating stability
    [1, 1, 0, 1],  # yellow: 3 = object directly above violation
    [0, 0, 1, 1],  # blue: 4 = upper part of the stack (unstable)
    # NOT USED!
    [0, 1, 1, 1],  # cyan
    [1, 0, 1, 1],  # magenta
    [1, 1, 1, 1],  # white : unassigned pixel
]

# color codes for the instance segmentation maps; list index corresponds to class label
ISEG_COLOR_CODES = [
    [0, 0, 0, 1],  # black: background pixel
    [1, 0, 0, 1],  # red: shape #1
    [0, 1, 0, 1],  # green: shape #2
    [0, 0, 1, 1],  # blue: shape #3
    [1, 1, 0, 1],  # yellow: shape #4
    [0, 1, 1, 1],  # cyan: shape #5
    [1, 0, 1, 1],  # magenta: shape #6
    # NOT USED!
    [1, 1, 1, 1],  # white: unassigned pixel
]
