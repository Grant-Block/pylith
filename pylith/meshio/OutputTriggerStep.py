# ----------------------------------------------------------------------
#
# Brad T. Aagaard, U.S. Geological Survey
# Charles A. Williams, GNS Science
# Matthew G. Knepley, University at Buffalo
#
# This code was developed as part of the Computational Infrastructure
# for Geodynamics (http://geodynamics.org).
#
# Copyright (c) 2010-2022 University of California, Davis
#
# See LICENSE.md for license information.
#
# ----------------------------------------------------------------------

from .OutputTrigger import OutputTrigger
from .meshio import OutputTriggerStep as ModuleOutputTriggerStep


class OutputTriggerStep(OutputTrigger, ModuleOutputTriggerStep):
    """
    Define how often output is written in terms of solution steps.

    Implements `OutputTrigger`.
    """
    DOC_CONFIG = {
        "cfg": """
            [output_trigger]
            num_skip = 2
        """
    }

    import pythia.pyre.inventory

    numSkip = pythia.pyre.inventory.int("num_skip", default=0, validator=pythia.pyre.inventory.greaterEqual(0))
    numSkip.meta['tip'] = "Number of solution steps to skip between writes (0 means write every time step)."

    def __init__(self, name="outputtriggerstep"):
        """Constructor.
        """
        OutputTrigger.__init__(self, name)

    def preinitialize(self):
        """Setup output trigger.
        """
        ModuleOutputTriggerStep.__init__(self)
        ModuleOutputTriggerStep.setIdentifier(self, self.aliases[-1])
        ModuleOutputTriggerStep.setNumStepsSkip(self, self.numSkip)

    def _configure(self):
        """Set members based using inventory.
        """
        OutputTrigger._configure(self)

# FACTORIES ////////////////////////////////////////////////////////////


def output_trigger():
    """Factory associated with OutputTriggerStep.
    """
    return OutputTriggerStep()


# End of file
