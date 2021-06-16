# ----------------------------------------------------------------------
#
# Grant A. Block, University of New Mexico
# Mousumi Roy, University of New Mexico
#
# This code was developed as part of the Computational Infrastructure
# for Geodynamics (http://geodynamics.org).
#
# Copyright (c) 2010-2018 University of California, Davis
#
# See COPYING for license information.
#
# ----------------------------------------------------------------------
#
# @file tests/fullscale/poroelasticity/point_source/injection_soln.py
#
# @brief Analytical solution to Segall's problem in Ch. 7 of Wang

import numpy as np
import math

# Physical properties
rho_s = 2500  # kg / m**3
rho_f = 1000  # kg / m**3
mu_f = 1.0  # Pa*s
G = 3.0  # Pa
K_sg = 10.0  # Pa
K_fl = 8.0  # Pa
K_d = 4.0  # Pa
# K_u = 2.6941176470588233 # Pa
alpha = 0.6  # -
phi = 0.1  # -
# M = 4.705882352941176# Pa
k = 1.5  # m**2


ymax = 5.0e3  # m
ymin = 0.0  # m
xmax = -2.0e3  # m
xmin = 2.0e3  # m
P_0 = -1.0  # Pa

# Problem Dimensions
L   = 1e4 # finite line length in m
b   = 100 # layer thickness in m
D   = 1000 # source depth in m

M = 1.0 / (phi / K_fl + (alpha - phi) / K_sg)  # Pa
K_u = K_d + alpha * alpha * M  # Pa,      Cheng (B.5)
nu = (3.0 * K_d - 2.0 * G) / (2.0 * (3.0 * K_d + G))  # -,       Cheng (B.8)
nu_u = (3.0 * K_u - 2.0 * G) / (2.0 * (3.0 * K_u + G))  # -,       Cheng (B.9 undrained poissons coeff
eta = (3.0 * alpha * G) / (3.0 * K_d + 4.0 * G)  # -,       Cheng (B.11)
S = (3.0 * K_u + 4.0 * G) / (M * (3.0 * K_d + 4.0 * G))  # Pa^{-1}, Cheng (B.14)
c = (k / mu_f) / S  # m^2 / s, Cheng (B.16)
B = 0.6 # Skempton's coefficient
V_0_dot = 2.0e+6/3.0e+07 # in m^3/s
Q_0 = -V_0_dot/(L*b)
prefac = 2 * B * (1 + nu_u) * Q_0 * b * D / (3*np.pi) #prefactor for displacement calculation

ts = 0.0028666667  # sec
nts = 2
tsteps = np.arange(0.0, ts * nts, ts)  + ts # sec

class AnalyticalSoln(object):
    """Analytical solution to Segall's injection problem
    """

    SPACE_DIM = 2
    TENSOR_SIZE = 4
    ITERATIONS = 16000

    def __init__(self):
        self.fields = {
            "displacement": self.displacement,
            "pressure": self.pressure,
            "porosity": self.porosity,
            "trace_strain": self.trace_strain,
            "solid_density": self.solid_density,
            "fluid_density": self.fluid_density,
            "fluid_viscosity": self.fluid_viscosity,
            "shear_modulus": self.shear_modulus,
            "drained_bulk_modulus": self.drained_bulk_modulus,
            "biot_coefficient": self.biot_coefficient,
            "biot_modulus": self.biot_modulus,
            "isotropic_permeability": self.isotropic_permeability,
            "initial_amplitude": {
                "x_neg": self.zero_vector,
                "x_pos": self.zero_vector,
                "y_pos_neu": self.y_pos_neu,
                "y_pos_dir": self.zero_scalar,
                "y_neg": self.zero_vector,
            }
        }
        self.key = None
        return

    def getField(self, name, pts):
        if self.key is None:
            field = self.fields[name](pts)
        else:
            field = self.fields[name][self.key](pts)
        return field

    def zero_scalar(self, locs):
        (npts, dim) = locs.shape
        return np.zeros((1, npts, 1), dtype=np.float64)

    def zero_vector(self, locs):
        (npts, dim) = locs.shape
        return np.zeros((1, npts, self.SPACE_DIM), dtype=np.float64)

    def solid_density(self, locs):
        """Compute solid_density field at locations.
        """
        (npts, dim) = locs.shape
        solid_density = rho_s * np.ones((1, npts, 1), dtype=np.float64)
        return solid_density

    def fluid_density(self, locs):
        """Compute fluid density field at locations.
        """
        (npts, dim) = locs.shape
        fluid_density = rho_f * np.ones((1, npts, 1), dtype=np.float64)
        return fluid_density

    def shear_modulus(self, locs):
        """Compute shear modulus field at locations.
        """
        (npts, dim) = locs.shape
        shear_modulus = G * np.ones((1, npts, 1), dtype=np.float64)
        return shear_modulus

    def porosity(self, locs):
        """Compute porosity field at locations.
        """
        (npts, dim) = locs.shape
        porosity = phi * np.ones((1, npts, 1), dtype=np.float64)
        return porosity

    def fluid_viscosity(self, locs):
        """Compute fluid_viscosity field at locations.
        """
        (npts, dim) = locs.shape
        fluid_viscosity = mu_f * np.ones((1, npts, 1), dtype=np.float64)
        return fluid_viscosity

    def drained_bulk_modulus(self, locs):
        """Compute undrained bulk modulus field at locations.
        """
        (npts, dim) = locs.shape
        undrained_bulk_modulus = K_d * np.ones((1, npts, 1), dtype=np.float64)
        return undrained_bulk_modulus

    def biot_coefficient(self, locs):
        """Compute biot coefficient field at locations.
        """
        (npts, dim) = locs.shape
        biot_coefficient = alpha * np.ones((1, npts, 1), dtype=np.float64)
        return biot_coefficient

    def biot_modulus(self, locs):
        """Compute biot modulus field at locations.
        """
        (npts, dim) = locs.shape
        biot_modulus = M * np.ones((1, npts, 1), dtype=np.float64)
        return biot_modulus

    def isotropic_permeability(self, locs):
        """Compute isotropic permeability field at locations.
        """
        (npts, dim) = locs.shape
        isotropic_permeability = k * np.ones((1, npts, 1), dtype=np.float64)
        return isotropic_permeability

    def ierfc(self, z):
        """Integrated error function
        """
        return np.exp(-z**2)/np.sqrt(np.pi) - z*(1 - math.erf(z))

    def integral(self, locs, x, t):
        """Integral for eq. 7.134 of Wang
        """
        (npts, dim) = locs.shape
        xi = np.linspace(xmin, xmax, npts)
        argum = np.sqrt(xi**2/(4*c*t))
        denom = D**2 + (x - xi)**2
        integrand = self.ierfc(argum)/denom

        return np.trapz(integrand, dx=0.1)

    def displacement(self, locs):
        """Calculate displacement with eq. 7.134 of Wang
        """

        (npts, dim) = locs.shape
        ntpts = tsteps.shape[0]
        displacement = np.zeros((ntpts, npts, dim), dtype=np.float64)
        x = locs[:, 0]
        t_track = 0

        for t in tsteps:
            displacement[t_track, :, 1 ] = prefac*self.integral(locs, x, t) 
        return displacement
