import numpy as np
import pandas as pd
from math import floor, ceil
from efficiency_functions import base_eff, sample_params

seed = 123
save_folder = './input_data'
N_DATASETS = 10
N_CHARGE_STATES = 5
N_LINES = 200
WL_MIN, WL_MAX = 2, 30
TRAIN_PERCENT = 0.8


rng = np.random.default_rng(seed=seed)

test_wls = np.linspace(0, 30, 100)
true_effs = {'wavelength': test_wls}

for i in range(N_DATASETS):
    eff_params = sample_params(rng)
    for mode in ['train', 'test']:
        if mode == 'train':
            n_lines_mode = floor(N_LINES * TRAIN_PERCENT)
        elif mode == 'test':
            n_lines_mode = floor(N_LINES * (1 - TRAIN_PERCENT))
        else:
            raise Exception('Invalid Mode')

        charge_states = rng.choice(range(N_CHARGE_STATES), n_lines_mode)
        charge_state_intensities = rng.uniform(0.5, 1.5, N_CHARGE_STATES)

        wls = rng.uniform(WL_MIN, WL_MAX, n_lines_mode)
        theoretical_intensities = 10 ** rng.uniform(2, 4, n_lines_mode)

        true_effs[i] = base_eff(test_wls, **eff_params)

        experimental_intensities = theoretical_intensities * base_eff(wls, **eff_params)
        experimental_uncertainties = rng.uniform(np.sqrt(experimental_intensities) * 0.9,
                                                 np.sqrt(experimental_intensities) * 1.1)
        experimental_intensities += rng.normal(0, experimental_uncertainties)

        theoretical_intensities *= charge_state_intensities[charge_states]

        data_dict = {'ChargeState':charge_states, 'Wavelength': wls, 'ExpIntensity':experimental_intensities,
                     'ExpIntensityUnc': experimental_uncertainties, 'TheoryIntensity':theoretical_intensities}
        df = pd.DataFrame(data_dict)
        df.to_csv(save_folder + f'/dataset{i}_{mode}.csv', index=False)

pd.DataFrame(true_effs).to_csv('true_efficiencies.csv', index=False)