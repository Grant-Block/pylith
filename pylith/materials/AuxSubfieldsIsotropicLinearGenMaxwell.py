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

from pylith.utils.PetscComponent import PetscComponent


class AuxSubfieldsIsotropicLinearGenMaxwell(PetscComponent):
    """
    Auxiliary subfields associated with the isotropic generalized Maxwell viscoelastic bulk rheology.
    """
    DOC_CONFIG = {
        "cfg": """
            [pylithapp.problem.materials.mat_genmaxwell.rheology.auxiliary_fields]
            shear_modulus.basis_order = 1
            bulk_modulus.basis_order = 1
            maxwell_time.basis_order = 1
            shear_modulus_ratio.basis_order = 1
            total_strain.basis_order = 1
            viscous_strain.basis_order = 1
            reference_stress.basis_order = 0
            reference_strain.basis_order = 0
        """
    }

    import pythia.pyre.inventory
    from pylith.topology.Subfield import Subfield

    shearModulus = pythia.pyre.inventory.facility("shear_modulus", family="auxiliary_subfield", factory=Subfield)
    shearModulus.meta['tip'] = "Shear modulus subfield."

    bulkModulus = pythia.pyre.inventory.facility("bulk_modulus", family="auxiliary_subfield", factory=Subfield)
    bulkModulus.meta['tip'] = "Bulk modulus subfield."

    maxwellTime = pythia.pyre.inventory.facility("maxwell_time", family="auxiliary_subfield", factory=Subfield)
    maxwellTime.meta['tip'] = "Maxwell time subfield for 3 Maxwell elements."

    shearModulusRatio = pythia.pyre.inventory.facility("shear_modulus_ratio", family="auxiliary_subfield", factory=Subfield)
    shearModulusRatio.meta['tip'] = "Shear modulus ratio subfield for 3 Maxwell elements."

    totalStrain = pythia.pyre.inventory.facility("total_strain", family="auxiliary_subfield", factory=Subfield)
    totalStrain.meta['tip'] = "Total strain subfield."

    viscousStrain = pythia.pyre.inventory.facility("viscous_strain", family="auxiliary_subfield", factory=Subfield)
    viscousStrain.meta['tip'] = "Viscous strain subfield for 3 Maxwell elements."

    referenceStress = pythia.pyre.inventory.facility("reference_stress", family="auxiliary_subfield", factory=Subfield)
    referenceStress.meta['tip'] = "Reference stress subfield."

    referenceStrain = pythia.pyre.inventory.facility("reference_strain", family="auxiliary_subfield", factory=Subfield)
    referenceStrain.meta['tip'] = "Reference strain subfield."

    def __init__(self, name="auxfieldsisotropiclineargenmaxwell"):
        """Constructor.
        """
        PetscComponent.__init__(self, name, facility="auxiliary_subfields")

    def _configure(self):
        PetscComponent._configure(self)


# FACTORIES ////////////////////////////////////////////////////////////

def auxiliary_subfields():
    """Factory associated with AuxSubfieldsIsotropicLinearGenMaxwell.
    """
    return AuxSubfieldsIsotropicLinearGenMaxwell()


# End of file
