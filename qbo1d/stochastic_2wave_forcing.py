import numpy as np
import torch

from . import utils


def sample_as_cs(n, ase, asv, cse, csv, corr, seed):
    """Draw a squence of source fluxes and spectral widths.

    The wave amplitudes and phase speeds are first drawn from a bivariate
    normal distribution with the specified correlation and then mapped
    to bivariate log-normal distribution with the specified means and variances.

    Parameters
    ----------
    n : int
        Number of realizations
    ase : float
        Wave amplitude mean
    asv : float
        Wave amplitude variance
    cse : float
        Phase speed mean
    csv : float
        Phase speed variance
    corr : float
        Correlation in the underlying normal distribution
    seed : int
        A seed for the pseudorandom number generator

    Returns
    -------
    (float, float)
        Amplitudes, phase speeds
    """

    # means and variances of the bivariate log-normal distribution
    es = torch.tensor([ase, cse])
    vs = torch.tensor([asv, csv])

    # resulting means and variances of the corresponding normal distribution
    mu = - 0.5 * torch.log(vs / es**4 + 1 / es**2)
    variances = torch.log(es**2) - 2 * mu

    # covariance matrix
    sigma = torch.tensor([[variances[0],
    corr*(variances[0]*variances[1])**0.5],
    [corr*(variances[0]*variances[1])**0.5,
    variances[1]]])

    # choose seed for reproducibility
    torch.manual_seed(seed)

    # draw from normal distribution
    normal_dist = (
    torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma))
    normal_samples = normal_dist.sample((n,))

    # map to log-normal distribution
    lognormal_samples = torch.exp(normal_samples)

    # amplitudes and phase speeds
    As = lognormal_samples[:, 0]
    cs = lognormal_samples[:, 1]

    return As, cs


class WaveSpectrum(torch.nn.Module):
    """A ModelClass for setting up the stochastic source term.

    Parameters
    ----------
    solver : ADSolver
        A solver instance holding the grid and differentiation matrix
    Gsa : float, optional
        Amplitude of semi-annual oscillation [:math:`\mathrm{m \, s^{-2}}`], by default 0
    ase : float, optional
        Wave amplitude mean, by default 6.0e-4 / 0.1006
    asv : float, optional
        Wave amplitude variance, by default 1e-12
    cse : float, optional
        Phase speed mean, by default 32
    csv : float, optional
        Phase speed variance, by default 25
    corr : float, optional
        Correlation in the underlying normal distribution, by default 0.75
    seed : int, optional
        A seed for the pseudorandom number generator, by default 197

    Attributes
    ----------
    g_func : func
        An interface for keeping track of the function g in the analytic forcing
    F_func : func
        An interface for keeping track of the function F in the analytic forcing
    G_func : func
        An interface for keeping track of the semi-annual oscillation
    ks : tensor
        Wavenumbers
    cs : tensor
        Phase speeds
    As : tensor
        Wave amplitudes
    """

    def __init__(self, solver, Gsa=0,
        ase=6.e-4, asv=1e-12, cse=32, csv=25, corr=0.75, seed=int(21*9+8)):
        super().__init__()

        self.train(False)

        self._z = solver.z
        self._nlev = solver.nlev
        self._nsteps, = solver.time.shape
        self._D1 = solver.D1

        self._rho = utils.get_rho(self._z)
        self._alpha = utils.get_alpha(self._z)

        self._current_step = 0
        self._current_time = solver.current_time

        # keep track of source
        self.s = torch.zeros((self._nsteps, self._nlev))

        # sample amplitudes and phase speeds
        As, cs = sample_as_cs(n=self._nsteps,
        ase=ase, asv=asv, cse=cse, csv=csv, corr=corr, seed=seed)

        # scale amplitudes by rho_0
        As /= 0.1006

        # wave amplitudes and phase speeds
        self.As = torch.transpose(torch.vstack([As, -As]), 0, 1)
        self.cs = torch.transpose(torch.vstack([cs, -cs]), 0, 1)

        print(self.As.shape)

        # wavenumbers
        self.ks = 1 * 2 * torch.pi / 4e7 * torch.ones(2)
        print(self.ks.shape)

        self.g_func = lambda c, k, u : (utils.NBV * self._alpha /
        (k * ((c - u) ** 2)))

        self.F_func = lambda A, g : (A * torch.exp(-torch.hstack((
            torch.zeros(1),
            torch.cumulative_trapezoid(g, dx=solver.dz)
            ))))

        self.G_func = lambda z, t : torch.where(
            (28e3 <= z) & (z <= 35e3),
            (Gsa * 2 * (z - 28e3) * 1e-3 * 2 * torch.pi / 180 / 86400 *
            torch.sin(2 * torch.pi / 180 / 86400 * t)),
            torch.zeros(1))

    def forward(self, u):
        """An interface for calculating the source term as a function of u. By
        default, torch.nn.Module uses the forward method.

        Parameters
        ----------
        u : tensor
            Zonal wind profile

        Returns
        -------
        tensor
            Source term as a function of the zonal wind u
        """

        Ftot = torch.zeros(u.shape)
        for A, c, k in zip(self.As[self._current_step, :], self.cs[self._current_step, :], self.ks):
            g = self.g_func(c, k, u)
            F = self.F_func(A, g)
            Ftot += F

        G = self.G_func(self._z, self._current_time)

        s = torch.matmul(self._D1, Ftot) * self._rho[0] / self._rho - G
        self.s[self._current_step, :] = s

        self._current_step += 1

        return s

