"""A class for abstracting the files associated with a particular model.

"""
import os
import re
from dataclasses import dataclass, field
from typing import Union

import cftime as cf
import numpy as np
import pandas as pd
import xarray as xr
from sympy import sympify

from Variable import Variable


def same_space_grid_size(vrs: dict[Variable]) -> bool:
    """
    Do all the variables have the same spatial grid sizes?
    """
    vref = vrs[next(iter(vrs))]
    if not np.all(
        [vrs[v].ds[vrs[v].lat_name].size == vref.ds[vref.lat_name].size for v in vrs]
    ):
        return False
    if not np.all(
        [vrs[v].ds[vrs[v].lon_name].size == vref.ds[vref.lon_name].size for v in vrs]
    ):
        return False
    return True


def compute_multimodel_mean(
    vrs: dict[Variable], grid_resolution: float = 1.0
) -> Variable:
    """
    Given a dictionary of variables, computes the mean. This function can be
    used to compute the mean from a model's ensemble or a mean from a multimodel
    group.

    * if the variables are temporal, we will redefine the time dimension to use
      a 'noleap' calendar
    * if the spatial grids are not uniformly the same size, we will interpolate
      to a uniformly spaced grid of size 'grid_resolution'

    Potential problems and improvements:

    * the interpolated grid is set to use longitude on [0,360] which will be
      problematic if combining models which use a different convention
    * because we compute the mean by adding a 'model' dimension using
      xarray.concat, we could implement a standard deviation also and/or a count
      of contributing models as well
    * units and variable names are not checked, we assume you have passed us a
      sensible set of variables to combine
    """
    vref = vrs[next(iter(vrs))]
    dss = [vrs[v].ds for v in vrs]
    if vref.temporal():
        tbnd = []
        for var in vrs:
            vrs[var].ds["time"] = [
                cf.DatetimeNoLeap(t.dt.year, t.dt.month, t.dt.day)
                for t in vrs[var].ds["time"]
            ]
            tbnd.append(vrs[var].timeBounds())
        init_t = max([t[0] for t in tbnd])
        final_t = min([t[1] for t in tbnd])
        init_t = cf.DatetimeNoLeap(init_t.dt.year, init_t.dt.month, 1)
        final_t = cf.DatetimeNoLeap(final_t.dt.year, final_t.dt.month, 28)
        dss = [ds.sel(time=slice(init_t, final_t)) for ds in dss]
    if vref.spatial() and not same_space_grid_size(vrs):
        lat = np.linspace(-90, 90, int(round(180 / grid_resolution)) + 1)
        lon = np.linspace(0, 360, int(round(360 / grid_resolution)) + 1)
        lat = 0.5 * (lat[:-1] + lat[1:])
        lon = 0.5 * (lon[:-1] + lon[1:])
        dss = [
            ds.interp(coords={"lat": lat, "lon": lon}, method="nearest") for ds in dss
        ]
    dss = [ds[vref.varname] for ds in dss]
    dss = xr.align(*dss, join="override")
    dss = xr.concat(dss, dim="model").mean(dim="model")
    mean = Variable(
        da=dss, varname=vref.varname, time_measure=compute_time_measure(dss[0])
    )
    mean.ds[mean.varname].attrs = vref.ds[vref.varname].attrs
    return mean


def compute_time_measure(vds: xr.Dataset) -> xr.DataArray:
    """
    this is duplicate code from inside Variable
    """
    tms = None
    tb_name = None
    if "bounds" in vds["time"].attrs:
        tb_name = vds["time"].attrs["bounds"]
    if tb_name is not None and tb_name in vds:
        tms = vds[tb_name]
        nmb = tms.dims[-1]
        tms = tms.diff(nmb).squeeze()
        tms *= 1e-9 / 86400  # [ns] to [d]
        tms = tms.astype("float")
    return tms


class VarNotInModel(Exception):
    """A exception to indicate that a variable is not present in the model
    results."""

    def __init__(self, variable: str, model: "ModelResult"):
        super().__init__(f"{variable} not found in {model.name}")


@dataclass
class ModelResult:
    """A class for abstracting and managing model results."""

    name: str
    color: tuple[float] = (0, 0, 0)
    children: dict = field(init=False, default_factory=dict)
    synonyms: dict = field(init=False, repr=False, default_factory=dict)
    variables: dict = field(init=False, repr=False, default_factory=dict)
    area_atm: xr.DataArray = field(init=False, repr=False, default_factory=lambda: None)
    area_ocn: xr.DataArray = field(init=False, repr=False, default_factory=lambda: None)
    frac_lnd: xr.DataArray = field(init=False, repr=False, default_factory=lambda: None)
    results: pd.DataFrame = field(init=False, repr=False, default_factory=lambda: None)

    def _by_regex(self, group_regex: str) -> dict:
        """Create a partition of the variables by regex"""
        groups = {}
        for var, files in self.variables.items():
            to_remove = []
            for filename in files:
                match = re.search(group_regex, filename)
                if match:
                    grp = match.group(1)
                    if grp not in groups:
                        groups[grp] = {}
                    if var not in groups[grp]:
                        groups[grp][var] = []
                    groups[grp][var].append(filename)
                    to_remove.append(filename)
            for rem in to_remove:
                self.variables[var].remove(rem)  # get rid of those we passed to models
        return groups

    def _by_attr(self, group_attr: str) -> dict:
        """Create a partition of the variables by attr"""
        groups = {}
        for var, files in self.variables.items():
            to_remove = []
            for filename in files:
                with xr.open_dataset(filename) as dset:
                    if group_attr in dset.attrs:
                        grp = dset.attrs[group_attr]
                        if grp not in groups:
                            groups[grp] = {}
                        if var not in groups[grp]:
                            groups[grp][var] = []
                        groups[grp][var].append(filename)
                        to_remove.append(filename)
            for rem in to_remove:
                self.variables[var].remove(rem)  # get rid of those we passed to models
        return groups

    def find_files(
        self,
        path: Union[str, list[str]],
        child_regex: str = None,
        child_attr: str = None,
    ):
        """Given a path or list of paths, find all netCDF files and variables
        therein. If 'group_regex' or 'group_attr' is given, then this function
        will create child models based on either a match with a regular
        expression or an attribute in the global attributes of the file.
        """
        if isinstance(path, str):
            path = [path]
        for file_path in path:
            for root, _, files in os.walk(file_path, followlinks=True):
                for filename in files:
                    if not filename.endswith(".nc"):
                        continue
                    filepath = os.path.join(root, filename)
                    with xr.open_dataset(filepath) as dset:
                        for key in dset.variables.keys():
                            if key not in self.variables:
                                self.variables[key] = []
                            self.variables[key].append(filepath)

        # create sub-models automatically in different ways
        groups = {}
        if not groups and child_regex is not None:
            groups = self._by_regex(child_regex)
        if not groups and child_attr is not None:
            groups = self._by_attr(child_attr)
        for grp, submodel in groups.items():
            mod = ModelResult(name=grp)
            mod.variables = submodel
            self.children[mod.name] = mod
        return self

    def get_variable(
        self,
        vname: str,
        synonyms: Union[str, list[str]] = None,
        mean: bool = False,
        **kwargs: dict,
    ) -> Union[Variable, dict[Variable]]:
        """Search the model database for the specified variable."""
        if not self.children:
            return self._get_variable_child(
                vname, synonyms=synonyms, mean=mean, **kwargs
            )
        out = {}
        for child, mod in self.children.items():
            try:
                # pylint: disable=protected-access
                out[child] = mod._get_variable_child(
                    vname, synonyms=synonyms, mean=mean, **kwargs
                )
            except VarNotInModel:
                pass
        if len(out) == 0:
            raise ValueError(f"Cannot find '{vname}' in '{self.name}'")
        if mean:
            vmean = compute_multimodel_mean(out)
            out["mean"] = vmean
            return out
        return out

    def _get_variable_child(
        self,
        vname: str,
        synonyms: Union[str, list[str]] = None,
        **kwargs: dict,
    ) -> Union[Variable, dict[Variable]]:
        """Get the model variable out of a single child model."""
        possible = [vname]
        if isinstance(synonyms, str):
            possible.append(synonyms)
        elif isinstance(synonyms, list):
            possible += synonyms
        for pos in possible:
            if pos in self.synonyms:
                possible.append(self.synonyms[pos])
        for var in possible:
            if var not in self.variables:
                continue
            return self._get_variable_no_syn(var, **kwargs)
        raise VarNotInModel(vname, self)

    def _get_variable_no_syn(self, vname: str, **kwargs: dict):
        """At some point we need a routine to get the data from the model
        files without the possibility of synonyms or we end up in an infinite
        recursion. This is where we should do all the trimming / checking of
        sites, etc.
        """
        # Even if multifile, we need to peak at attributes from one file to get
        # cell measure information.
        files = sorted(self.variables[vname])
        if len(files) == 0:
            raise VarNotInModel(vname, self)
        dset = xr.open_dataset(files[0])
        var = dset[vname]
        area = None
        if "cell_measures" in var.attrs and vname not in ["areacella", "areacello"]:
            if "areacella" in var.attrs["cell_measures"] and self.area_atm is not None:
                area = self.area_atm.copy()
            if "areacello" in var.attrs["cell_measures"] and self.area_ocn is not None:
                area = self.area_ocn.copy()
            if "cell_methods" in var.attrs and area is not None:
                if (
                    "where land" in var.attrs["cell_methods"]
                    and self.frac_lnd is not None
                ):
                    area *= self.frac_lnd

        # initialize the variable with time trimming
        init_t = kwargs.get("t0", None)
        final_t = kwargs.get("tf", None)
        if len(files) == 1:
            var = Variable(
                filename=files[0],
                varname=vname,
                cell_measure=area,
                t0=init_t,
                tf=final_t,
            )
        else:
            var = [xr.open_dataset(v).sel(time=slice(init_t, final_t)) for v in files]
            var = sorted([v for v in var if v.time.size > 0], key=lambda a: a.time[0])
            var = xr.concat(var, dim="time")
            var = Variable(
                da=var[vname],
                varname=vname,
                cell_measure=area,
                time_measure=compute_time_measure(var),
            )
        # This is where we would put site extraction?
        return var

    def get_grid_information(self):
        """If part of the model, grab the atmosphere and ocean cell areas as
        well as the land fractions. When a variable is obtained from the model,
        these measures are passed along along based on the 'cell_measures'
        attribute in the netCDF file."""
        atm = ocn = lnd = None
        try:
            atm = self._get_variable_child("areacella", synonyms="area").convert("m2")
            atm = atm.ds[atm.varname]
        except VarNotInModel:
            pass
        try:
            ocn = self._get_variable_child("areacello").convert("m2")
            ocn = ocn.ds[ocn.varname]
        except VarNotInModel:
            pass
        try:
            lnd = self._get_variable_child("sftlf", synonyms="landfrac").convert("1")
            lnd = lnd.ds[lnd.varname]
        except VarNotInModel:
            pass
        if atm is not None and lnd is not None:
            # pylint: disable=unbalanced-tuple-unpacking
            atm, lnd = xr.align(atm, lnd, join="override", copy=False)
        if atm is not None:
            self.area_atm = atm
        if ocn is not None:
            self.area_ocn = ocn
        if lnd is not None:
            self.frac_lnd = lnd
        for child, _ in self.children.items():
            self.children[child].get_grid_information()

    def add_synonym(self, expr):
        """Add synonyms that will be global for any variable or expression
        pulled from this model."""
        assert isinstance(expr, str)
        if expr.count("=") != 1:
            raise ValueError("Add a synonym by providing a string of the form 'a = b'")
        key, expr = expr.split("=")
        # check that the free symbols of the expression are variables
        for arg in sympify(expr).free_symbols:
            assert arg.name in self.variables
        self.synonyms[key] = expr

    def add_model(self, mod: Union["ModelResult", list["ModelResult"]]):
        """Add a child model to this model."""
        if isinstance(mod, ModelResult):
            mod = [mod]
        for child in mod:
            if child.name not in self.children:
                self.children[child.name] = child


if __name__ == "__main__":

    def test_model_ensemble():
        """tests a model ensemble"""
        mod = ModelResult(name="CESM2")
        mod.find_files(
            "/home/nate/data/ILAMB/MODELS/CESM/",
            child_regex=r"(r\d+i\d+p\d+f\d+)",
        )
        mod.get_grid_information()
        var = mod.get_variable("gpp", t0="1980-01-01", tf="2000-01-01", mean=True)
        scalar = (
            var["mean"]
            .integrate(dim="time", mean=True)
            .integrate(dim="space")
            .convert("Pg yr-1")
        )
        assert np.allclose(scalar.ds[scalar.varname], 128.8194)

    def test_single_model():
        """tests a single model"""
        mod = ModelResult(name="CESM2")
        mod.find_files("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2")
        mod.get_grid_information()
        mod.add_synonym("GPP=gpp")
        var = mod.get_variable("GPP", t0="1980-01-01", tf="2000-01-01")
        scalar = (
            var.integrate(dim="time", mean=True)
            .integrate(dim="space")
            .convert("Pg yr-1")
        )
        assert np.allclose(scalar.ds[scalar.varname], 108.8057)

    test_single_model()
    test_model_ensemble()
