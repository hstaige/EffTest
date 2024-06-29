import numpy as np
import pandas as pd
from efficiency_functions import base_eff, sample_params

seed = 123
save_folder = './input_data'
N_DATASETS = 10
N_CHARGE_STATES = 5
N_LINES = 100
WL_MIN, WL_MAX = 2, 30

rng = np.random.default_rng(seed=seed)

for i in range(N_DATASETS):
    charge_states = rng.choice(range(N_CHARGE_STATES), N_LINES)
    charge_state_intensities = rng.uniform(0.5, 1.5, N_CHARGE_STATES)

    wls = rng.uniform(WL_MIN, WL_MAX, N_LINES)
    theoretical_intensities = 10 ** rng.uniform(2, 4, N_LINES)

    experimental_intensities = theoretical_intensities * base_eff(wls, **sample_params(rng))
    experimental_uncertainties = rng.uniform(np.sqrt(experimental_intensities) * 0.9,
                                             np.sqrt(experimental_intensities) * 1.1)
    experimental_intensities += rng.normal(0, experimental_uncertainties)

    theoretical_intensities *= charge_state_intensities[charge_states]

    data_dict = {'ChargeState':charge_states, 'Wavelength': wls, 'ExpIntensity':experimental_intensities,
                 'ExpIntensityUnc': experimental_uncertainties, 'TheoryIntensity':theoretical_intensities}
    df = pd.DataFrame(data_dict)
    df.to_csv(save_folder + f'/dataset{i}.csv', index=False)