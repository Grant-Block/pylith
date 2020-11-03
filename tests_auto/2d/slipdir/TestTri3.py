#!/usr/bin/env python
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
# Copyright (c) 2010-2017 University of California, Davis
#
# See COPYING for license information.
#
# ----------------------------------------------------------------------
#

## @file tests/2d/slipdir/TestTri3.py
##
## @brief Generic tests for problems using 2-D mesh.

import unittest
import numpy

from pylith.tests import has_h5py

class TestTri3(unittest.TestCase):
  """
  Generic tests for problems using 2-D mesh.
  """

  def setUp(self):
    """
    Setup for tests.
    """
    self.mesh = {'ncells': 584,
                 'ncorners': 3,
                 'nvertices': 325,
                 'spaceDim': 2,
                 'tensorSize': 3}

    if has_h5py():
      self.checkResults = True
    else:
      self.checkResults = False
    return


  def test_elastic_statevars(self):
    """
    Check elastic state variables.
    """
    if not self.checkResults:
      return

    filename = "%s-statevars.h5" % self.outputRoot

    from pylith.tests.StateVariables import check_state_variables
    stateVars = ["total_strain", "stress"]
    check_state_variables(self, filename, self.mesh, stateVars)

    return


# End of file