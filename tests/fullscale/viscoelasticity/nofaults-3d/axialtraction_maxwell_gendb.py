#!/usr/bin/env nemesis
#
# ----------------------------------------------------------------------
#
# Brad T. Aagaard, U.S. Geological Survey
# Charles A. Williams, GNS Science
# Matthew G. Knepley, University of Chicago
#
# This code was developed as part of the Computational Infrastructure
# for Geodynamics (http://geodynamics.org).
#
# Copyright (c) 2010-2016 University of California, Davis
#
# See COPYING for license information.
#
# ----------------------------------------------------------------------
#
# @file tests/fullscale/viscoelasticity/nofaults-3d/axialtraction_maxwell_gendb.py
#
# @brief Python script to generate spatial database with displacement
# boundary conditions for the axial traction test. The traction boundary
# conditions use UniformDB in the .cfg file.

import numpy


class GenerateDB(object):
    """
    Python object to generate spatial database with displacement
    boundary conditions for the axial traction test.
    """

    def __init__(self):
        """
        Constructor.
        """
        return

    def run(self):
        """
        Generate the database.
        """
        # Domain
        x = numpy.arange(-4000.0, 4000.1, 2000.0)
        y = numpy.arange(-4000.0, 4000.1, 2000.0)
        z = numpy.arange(-8000.0, 0.1, 2000.0)
        (x3, y3, z3) = numpy.meshgrid(x, y, z)
        nptsX = x.shape[0]
        nptsY = y.shape[0]
        nptsZ = z.shape[0]

        xyz = numpy.zeros((nptsX * nptsY * nptsZ, 3), dtype=numpy.float64)
        xyz[:, 0] = x3.ravel()
        xyz[:, 1] = y3.ravel()
        xyz[:, 2] = z3.ravel()

        from axialtraction_maxwell_soln import AnalyticalSoln
        soln = AnalyticalSoln()
        disp = soln.displacement(xyz)

        from spatialdata.geocoords.CSCart import CSCart
        cs = CSCart()
        cs.inventory.spaceDim = 3
        cs._configure()
        data = {
            'x': x,
            'y': y,
            'z': z,
            'points': xyz,
            'coordsys': cs,
            'data_dim': 3,
            'values': [{'name': "initial_amplitude_x",
                        'units': "m",
                        'data': disp[0, :, 0].ravel()},
                       {'name': "initial_amplitude_y",
                        'units': "m",
                        'data': disp[0, :, 1].ravel()},
                       {'name': "initial_amplitude_z",
                        'units': "m",
                        'data': disp[0, :, 2].ravel()},
                       ]}

        from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
        io = SimpleGridAscii()
        io.inventory.filename = "axialtraction_maxwell_bc.spatialdb"
        io._configure()
        io.write(data)
        return


# ======================================================================
if __name__ == "__main__":
    GenerateDB().run()


# End of file