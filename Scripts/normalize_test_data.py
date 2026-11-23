import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_test_singles(dataset, scalers):
    """
    Normalize test data for single image lenses using pre‑fitted scalers.
    No time delay is used (always zero).
    """
    inputs = []
    outputs = []

    single_systems = [s for s in dataset if s['img'].shape[0] == 1]

    img_scaler = scalers['img_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']

    for system in single_systems:
        img_normalized = img_scaler.transform(system['img']).flatten()
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()

        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()

        # Input: 2 (img) + 2 (z) = 4
        input_vec = np.concatenate([img_normalized, z_normalized])
        # Output: 1 (potent) + 2 (deflec) = 3
        output_vec = np.concatenate([potent_normalized, deflec_normalized])

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs)

def normalize_test_doubles(dataset, scalers):
    """
    Normalize test data for double image lenses using pre‑fitted scalers.
    Only the non‑zero time delay (Δt) is used.
    """
    inputs = []
    outputs = []

    double_systems = [s for s in dataset if s['img'].shape[0] == 2]

    img_scaler = scalers['img_scaler']
    time_scaler = scalers['time_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']

    for system in double_systems:
        img_normalized = img_scaler.transform(system['img']).flatten()

        time_data = system['time'].to_value()[1:]   # skip zero
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()

        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()

        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()

        # Input: 4 (img) + 1 (time) + 2 (z) = 7
        input_vec = np.concatenate([img_normalized, time_normalized, z_normalized])
        # Output: 2 (potent) + 4 (deflec) = 6
        output_vec = np.concatenate([potent_normalized, deflec_normalized])

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs)

def normalize_test_quads(dataset, scalers):
    """
    Normalize test data for quad image lenses using pre‑fitted scalers.
    Only the three non‑zero time delays are used.
    """
    inputs = []
    outputs = []

    quad_systems = [s for s in dataset if s['img'].shape[0] == 4]

    img_scaler = scalers['img_scaler']
    time_scaler = scalers['time_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']

    for system in quad_systems:
        img_normalized = img_scaler.transform(system['img']).flatten()

        time_data = system['time'].to_value()[1:]   # skip zero
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()

        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()

        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()

        # Input: 8 (img) + 3 (time) + 2 (z) = 13
        input_vec = np.concatenate([img_normalized, time_normalized, z_normalized])
        # Output: 4 (potent) + 8 (deflec) = 12
        output_vec = np.concatenate([potent_normalized, deflec_normalized])

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs)

# The legacy normalize_test() function is removed because it is incompatible
# with the new per‑type scalers and input shapes.
# Use the specific functions above.