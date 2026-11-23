import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_single_lens_data(dataset):
    """Extract data for single image lenses (1 image) – NO time delay (always zero)"""
    inputs = []
    outputs = []

    single_systems = [s for s in dataset if s['img'].shape[0] == 1]

    # Initialize scalers (no time_scaler needed)
    img_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()

    # First pass: collect all data for fitting scalers
    all_images = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []

    for system in single_systems:
        all_images.append(system['img'])
        all_redshifts.append([system['zLens'], system['zSrc']])
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)
        all_deflections.append(system['deflec'])

    # Fit scalers
    if all_images:
        img_scaler.fit(np.vstack(all_images))
        z_scaler.fit(np.array(all_redshifts))
        potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))
        deflec_scaler.fit(np.vstack(all_deflections))

    # Second pass: normalize and process
    for system in single_systems:
        img_normalized = img_scaler.transform(system['img']).flatten()          # Shape: (2,)
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (1,)

        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()          # Shape: (2,)

        # Input: 2 (img) + 2 (z) = 4 features
        input_vec = np.concatenate([img_normalized, z_normalized])
        # Output: 1 (potent) + 2 (deflec) = 3 features
        output_vec = np.concatenate([potent_normalized, deflec_normalized])

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }

def extract_double_lens_data(dataset):
    """Extract data for double image lenses (2 images) – use only the non‑zero time delay (Δt)"""
    inputs = []
    outputs = []

    double_systems = [s for s in dataset if s['img'].shape[0] == 2]

    img_scaler = StandardScaler()
    time_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()

    all_images = []
    all_times = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []

    for system in double_systems:
        all_images.append(system['img'])

        # Keep only the non‑zero delay(s) – index 1 onward
        time_data = system['time'].to_value()[1:]   # shape (1,) for doubles
        all_times.append(time_data)

        all_redshifts.append([system['zLens'], system['zSrc']])

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)
        all_deflections.append(system['deflec'])

    if all_images:
        img_scaler.fit(np.vstack(all_images))
        time_scaler.fit(np.concatenate(all_times).reshape(-1, 1))
        z_scaler.fit(np.array(all_redshifts))
        potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))
        deflec_scaler.fit(np.vstack(all_deflections))

    for system in double_systems:
        img_normalized = img_scaler.transform(system['img']).flatten()     # Shape: (4,)

        time_data = system['time'].to_value()[1:]   # length 1
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (1,)

        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()   # Shape: (2,)

        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()           # Shape: (4,)

        # Input: 4 (img) + 1 (time) + 2 (z) = 7 features
        input_vec = np.concatenate([img_normalized, time_normalized, z_normalized])
        # Output: 2 (potent) + 4 (deflec) = 6 features
        output_vec = np.concatenate([potent_normalized, deflec_normalized])

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'time_scaler': time_scaler,
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }

def extract_quad_lens_data(dataset):
    """Extract data for quad image lenses (4 images) – use only non‑zero time delays (three Δt)"""
    inputs = []
    outputs = []

    quad_systems = [s for s in dataset if s['img'].shape[0] == 4]

    img_scaler = StandardScaler()
    time_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()

    all_images = []
    all_times = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []

    for system in quad_systems:
        all_images.append(system['img'])

        # Keep only the three non‑zero delays (skip the first)
        time_data = system['time'].to_value()[1:]   # shape (3,)
        all_times.append(time_data)

        all_redshifts.append([system['zLens'], system['zSrc']])

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)
        all_deflections.append(system['deflec'])

    if all_images:
        img_scaler.fit(np.vstack(all_images))
        time_scaler.fit(np.concatenate(all_times).reshape(-1, 1))
        z_scaler.fit(np.array(all_redshifts))
        potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))
        deflec_scaler.fit(np.vstack(all_deflections))

    for system in quad_systems:
        img_normalized = img_scaler.transform(system['img']).flatten()     # Shape: (8,)

        time_data = system['time'].to_value()[1:]   # length 3
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (3,)

        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)

        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()   # Shape: (4,)

        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()           # Shape: (8,)

        # Input: 8 (img) + 3 (time) + 2 (z) = 13 features
        input_vec = np.concatenate([img_normalized, time_normalized, z_normalized])
        # Output: 4 (potent) + 8 (deflec) = 12 features
        output_vec = np.concatenate([potent_normalized, deflec_normalized])

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'time_scaler': time_scaler,
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }