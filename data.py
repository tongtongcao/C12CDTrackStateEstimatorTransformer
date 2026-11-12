import numpy as np


# Fixed token order
LAYER_IDS = [1, 2, 3, 4, 7, 9, 5, 6, 8]

LAYER_TO_INDEX = {layer: i for i, layer in enumerate(LAYER_IDS)}


def read_tracks_with_crosses(filename):
    """
    Read a CSV file where each track is represented by two lines.

    Line 1:
        Flattened cross-level features.
        Each cross contains four values:

        SVT:  (x, y, z, layer)
        BMTC: (r, z, 0, layer)
        BMTZ: (r, phi, 0, layer)

    Line 2:
        Track state:
        (d0, phi0, kappa, z0, tandip)

    Returns
    -------
    features : np.ndarray
        Shape [N, 9, 3].
        Fixed 9-token input for Transformer.

    valid_mask : np.ndarray
        Shape [N, 9].
        True means the corresponding detector layer is present.

    states : np.ndarray
        Shape [N, 5].
        Target track states.
    """

    features_list = []
    masks_list = []
    states = []

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError("The file must contain two lines per track.")

    for i in range(0, len(lines), 2):

        # --------------------------------------------------
        # First line: detector crosses
        # --------------------------------------------------

        values = [float(x.strip()) for x in lines[i].split(",")]

        if len(values) % 4 != 0:
            raise ValueError(f"Line {i + 1}: cross data must contain a multiple of 4 values, got {len(values)}")

        crosses = np.asarray(values, dtype=np.float32).reshape(-1, 4)

        # Fixed 9-token representation
        X = np.zeros((9, 3), dtype=np.float32)

        # True = valid/present
        valid_mask = np.zeros(9, dtype=bool)

        for cross in crosses:
            v1, v2, v3, layer_value = cross
            layer = int(round(layer_value))

            if not np.isclose(layer_value, layer):
                raise ValueError(f"Invalid layer ID: {layer_value}")

            if layer not in LAYER_TO_INDEX:
                raise ValueError(f"Unexpected detector layer: {layer}")

            token = LAYER_TO_INDEX[layer]

            if valid_mask[token]:
                raise ValueError(f"Duplicate cross in layer {layer}, track index {i // 2}")

            # ----------------------------------------------
            # SVT: (x, y, z)
            # ----------------------------------------------
            if layer in [1, 2, 3]:
                X[token] = [v1, v2, v3]

            # ----------------------------------------------
            # BMTC: (r, z)
            # ----------------------------------------------
            elif layer in [4, 7, 9]:
                X[token] = [v1, v2, 0.0]

            # ----------------------------------------------
            # BMTZ: (r, phi)
            # Convert phi to sin(phi), cos(phi)
            # ----------------------------------------------
            elif layer in [5, 6, 8]:
                X[token] = [v1, np.sin(v2), np.cos(v2)]

            valid_mask[token] = True

        # Skip tracks without any valid cross
        if not valid_mask.any():
            continue

        features_list.append(X)
        masks_list.append(valid_mask)

        # --------------------------------------------------
        # Second line: target track state
        # --------------------------------------------------

        state_values = [float(x.strip()) for x in lines[i + 1].split(",")]

        if len(state_values) != 5:
            raise ValueError(f"Line {i + 2}: track state must contain exactly 5 values, got {len(state_values)}")

        states.append(state_values)

    features = np.asarray(features_list, dtype=np.float32)

    valid_mask = np.asarray(masks_list, dtype=bool)

    states = np.asarray(states, dtype=np.float32)

    return features, valid_mask, states


# --------------------------------------------------
# Example usage
# --------------------------------------------------

if __name__ == "__main__":

    features, valid_mask, states = (
        read_tracks_with_crosses("sample.csv")
    )

    print("Number of tracks:", len(features))

    print("Features shape:", features.shape)
    print("Mask shape:", valid_mask.shape)
    print("States shape:", states.shape)

    print("\nFirst track features:")
    print(features[0])

    print("\nFirst track valid mask:")
    print(valid_mask[0])

    print("\nFirst track target:")
    print(states[0])