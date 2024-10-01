import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from efficiency_functions import base_eff, sample_params

model_preds = pd.read_csv('model_preds.csv')
true_effs = pd.read_csv('true_efficiencies.csv')
pred_wl = np.linspace(0, 30, 300)
true_wl = true_effs['wavelength']

for i in range(10):
    true_eff = np.log(true_effs[str(i)])
    true_eff -= np.median(true_eff)
    pred_eff = model_preds[f'dataset{i}_mu']
    pred_eff -= np.median(pred_eff)
    std = model_preds[f'dataset{i}_std']

    f, ax = plt.subplots(figsize=(10,6))
    plt.plot(true_wl, true_eff, label='True log(eff)')
    plt.plot(pred_wl, pred_eff, label='Pred log(eff)')
    plt.fill_between(pred_wl, (pred_eff - std), (pred_eff + std), color='b', alpha=0.5, zorder=2)
    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(r'log($\nu$(eff))')
    plt.title(f'Dataset{i}')

plt.show()