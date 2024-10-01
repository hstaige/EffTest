import numpy as np
import matplotlib.pyplot as plt


def base_eff(x, const=0.4, lin=1e-2, lincen=20, quad=4e-4, sigamp=5e-1, sigcen=15, sigwidth=1):
    return const - np.abs((x-lincen) * lin) + x ** 2 * quad + sigamp / (1 + np.exp(-(x-sigcen)/sigwidth))


param_regions = {'const': (0.3, 0.6),
                 'lin': (5e-3, 2e-2),
                 'lincen': (5, 20),
                 'quad': (1e-4, 5e-4),
                 'sigamp': (3e-1, 7e-1),
                 'sigcen':(10, 20),
                 'sigwidth': (0.5, 3)}


def sample_params(rng, regions=param_regions):
    return {name: rng.uniform(low, high) for name, (low, high) in regions.items()}


if __name__ == '__main__':
    seed = 123
    rng = np.random.default_rng(seed=seed)

    for i in range(10):
        test_x = np.linspace(2, 30)
        test_y = base_eff(test_x, **sample_params(rng, param_regions))
        plt.plot(test_x, test_y)
        plt.show()