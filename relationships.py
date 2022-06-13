"""A class for developing and comparing relationships from gridded data.

This class was developed to help group and manage the data required to
generate and compare relationships. The module purposefully does not
depend on the ILAMB Variable object, but rather takes in the xarray
DataArray. This makes it more useful to users even outside this
codebase, but also means that the data arrays that you pass in must
have the units and regional masking already handled.

"""
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class Relationship:
    """A class for developing and comparing relationships from gridded data"""

    dep: xr.DataArray
    ind: xr.DataArray
    color: xr.DataArray = None
    dep_log: bool = False
    ind_log: bool = False
    dep_label: str = ""
    ind_label: str = ""
    order: int = 1
    _dep_limits: list[float] = field(init=False, default_factory=lambda: None)
    _ind_limits: list[float] = field(init=False, default_factory=lambda: None)
    _dist2d: np.ndarray = field(init=False, default_factory=lambda: None)
    _ind_edges: np.ndarray = field(init=False, default_factory=lambda: None)
    _dep_edges: np.ndarray = field(init=False, default_factory=lambda: None)
    _response_mean: np.ndarray = field(init=False, default_factory=lambda: None)
    _response_std: np.ndarray = field(init=False, default_factory=lambda: None)

    def __post_init__(self):

        # check input dataarrays for compatibility
        assert isinstance(self.dep, xr.DataArray)
        assert isinstance(self.ind, xr.DataArray)
        self.dep = self.dep.sortby(*self.dep.dims)
        self.ind = self.ind.sortby(*self.ind.dims)
        # pylint: disable=unbalanced-tuple-unpacking
        self.dep, self.ind = xr.align(self.dep, self.ind, join="exact")
        if self.color is not None:
            assert isinstance(self.color, xr.DataArray)
            self.color = self.color.sortby(*self.color)
            self.dep, self.ind, self.color = xr.align(
                self.dep, self.ind, self.color, join="exact"
            )

        # only consider where both are valid and finite
        keep = self.dep.notnull() * self.ind.notnull()
        keep *= np.isfinite(self.dep)
        keep *= np.isfinite(self.ind)
        self.dep = xr.where(keep, self.dep, np.nan)
        self.ind = xr.where(keep, self.ind, np.nan)
        if self.dep_log:
            assert self.dep.min() > 0
        if self.ind_log:
            assert self.ind.min() > 0

    def compute_limits(self, dep_lim=None, ind_lim=None):
        """Computes the limits of the dependent and independent variables.

        Parameters
        ----------
        dep_lim : array-like of size 2, optional
            if specified, will return the most extensive limits of the
            dependent variable and the input values
        ind_lim : array-like of size 2, optional
            if specified, will return the most extensive limits of the
            independent variable and the input values

        Returns
        -------
        dep_lim, ind_lim: array-like of size 2
            the most restrictive limits of the data and optional inputs
        """

        def _singlelimit(var, limit=None):
            lim = [var.min(), var.max()]
            delta = 1e-8 * (lim[1] - lim[0])
            lim[0] -= delta
            lim[1] += delta
            if limit is None:
                limit = lim
            else:
                limit[0] = min(limit[0], lim[0])
                limit[1] = max(limit[1], lim[1])
            return limit

        return _singlelimit(self.dep, dep_lim), _singlelimit(self.ind, ind_lim)

    def build_response(self, nbin=25, eps=3e-3):
        """Creates a 2D distribution and a functional response

        Parameters
        ----------
        nbin : int, optional
            the number of bins to use in both dimensions
        eps : float, optional
            the fraction of points required for a bin in the
            independent variable be included in the funcitonal responses
        """
        # if no limits have been created, make them now
        if self._dep_limits is None or self._ind_limits is None:
            self._dep_limits, self._ind_limits = self.compute_limits(
                dep_lim=self._dep_limits,
                ind_lim=self._ind_limits,
            )

        # compute the 2d distribution
        # pylint: disable=maybe-no-member
        ind = np.ma.masked_invalid(self.ind.values).compressed()
        dep = np.ma.masked_invalid(self.dep.values).compressed()
        xedges = nbin
        yedges = nbin
        if self.ind_log:
            xedges = 10 ** np.linspace(
                np.log10(self.ind_limits[0]), np.log10(self.ind_limits[1]), nbin + 1
            )
        if self.dep_log:
            yedges = 10 ** np.linspace(
                np.log10(self.dep_limits[0]), np.log10(self.dep_limits[1]), nbin + 1
            )
        dist, xedges, yedges = np.histogram2d(
            ind, dep, bins=[xedges, yedges], range=[self._ind_limits, self._dep_limits]
        )
        dist = np.ma.masked_values(dist.T, 0).astype(float)
        dist /= dist.sum()
        self._dist2d = dist
        self._ind_edges = xedges
        self._dep_edges = yedges

        # compute a binned functional response
        which_bin = np.digitize(ind, xedges).clip(1, xedges.size - 1) - 1
        mean = np.ma.zeros(xedges.size - 1)
        std = np.ma.zeros(xedges.size - 1)
        cnt = np.ma.zeros(xedges.size - 1)
        with np.errstate(under="ignore"):
            for i in range(mean.size):
                depi = dep[which_bin == i]
                cnt[i] = depi.size
                if self.dep_log:
                    depi = np.log10(depi)
                    mean[i] = 10 ** depi.mean()
                    std[i] = 10 ** depi.std()
                else:
                    mean[i] = depi.mean()
                    std[i] = depi.std()
            mean = np.ma.masked_array(mean, mask=(cnt / cnt.sum()) < eps)
            std = np.ma.masked_array(std, mask=(cnt / cnt.sum()) < eps)
        self._response_mean = mean
        self._response_std = std

    def plot_distribution(self, pax):
        """Plot the 2D histogram.

        Parameters
        ----------
        pax : matplotlib axis
            the axis on which to plot the function
        """
        if self._dist2d is None:
            self.build_response()
        dist_plot = pax.pcolormesh(
            self._ind_edges,
            self._dep_edges,
            self._dist2d,
            norm=LogNorm(vmin=1e-4, vmax=1e-1),
            cmap="plasma" if "plasma" in plt.cm.cmap_d else "summer",
        )
        div = make_axes_locatable(pax)
        pax.get_figure().colorbar(
            dist_plot,
            cax=div.append_axes("right", size="5%", pad=0.05),
            orientation="vertical",
            label="Fraction of total datasites",
        )
        pax.set_xlabel(self.ind_label, fontsize=12)
        pax.set_ylabel(self.dep_label, fontsize=12 if len(self.dep_label) <= 60 else 10)
        pax.set_xlim(self._ind_edges[0], self._ind_edges[-1])
        pax.set_ylim(self._dep_edges[0], self._dep_edges[-1])
        if self.dep_log:
            pax.set_yscale("log")
        if self.ind_log:
            pax.set_xscale("log")

    def plot_response(self, pax, color="k", shift=0):
        """Plot the mean response with standard deviation as error bars.

        Parameters
        ----------
        pax : matplotlib axis
            the axis on which to plot the function
        color : str, optional
            the color to use in plotting the line
        shift : float, optional
            shift expressed as a fraction on [-0.5,0.5] used to shift
            the plotting of the errorbars so that multiple functions do
            not overlab
        """
        if self._response_mean is None:
            self.build_response()
        r_mean = self._response_mean
        r_std = self._response_std
        ind_edges = self._ind_edges
        ind = 0.5 * (ind_edges[:-1] + ind_edges[1:]) + shift * np.diff(ind_edges).mean()
        mask = r_mean.mask
        if isinstance(mask, np.bool_):
            mask = np.asarray([mask] * r_mean.size)
        # pylint: disable=singleton-comparison
        ind = ind[mask == False]
        r_mean = r_mean[mask == False]
        r_std = r_std[mask == False]
        if self.dep_log:
            r_std = np.asarray(
                [
                    10 ** (np.log10(r_mean) - np.log10(r_std)),
                    10 ** (np.log10(r_mean) + np.log10(r_std)),
                ]
            )
        pax.errorbar(ind, r_mean, yerr=r_std, fmt="-o", color=color)
        pax.set_xlabel(self.ind_label, fontsize=12)
        pax.set_ylabel(self.dep_label, fontsize=12 if len(self.ind_label) <= 60 else 10)
        pax.set_xlim(self._ind_edges[0], self._ind_edges[-1])
        pax.set_ylim(self._dep_edges[0], self._dep_edges[-1])
        if self.dep_log:
            pax.set_yscale("log")
        if self.ind_log:
            pax.set_xscale("log")


# Simple test of the Relationship class using ILAMB data.
if __name__ == "__main__":
    import os

    ROOT = os.environ["ILAMB_ROOT"]
    gpp = xr.open_dataset(os.path.join(ROOT, "DATA/gpp/FLUXCOM/gpp.nc"))
    tas = xr.open_dataset(os.path.join(ROOT, "DATA/tas/CRU4.02/tas.nc"))

    with np.errstate(over="ignore"):
        gpp = gpp.mean(dim="time")["gpp"]
        tas = tas.mean(dim="time")["tas"]
    r = Relationship(
        gpp, tas, dep_label="FLUXCOM/gpp [g m-2 d-1]", ind_label="CRU4.02/tas [K]"
    )
    r.build_response()
    fig, ax = plt.subplots()
    r.plot_distribution(ax)
    r.plot_response(ax)
    plt.show()
