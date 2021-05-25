// -*- C++ -*-
//
// ======================================================================
//
// Brad T. Aagaard, U.S. Geological Survey
// Charles A. Williams, GNS Science
// Matthew G. Knepley, University of Chicago
//
// This code was developed as part of the Computational Infrastructure
// for Geodynamics (http://geodynamics.org).
//
// Copyright (c) 2010-2017 University of California, Davis
//
// See COPYING for license information.
//
// ======================================================================
//

/**
 * @file modulesrc/meshio/FieldFilterNone.i
 *
 * @brief Python interface to C++ FieldFilterNone object.
 */

namespace pylith {
    namespace meshio {
        class pylith::meshio::FieldFilterNone : public FieldFilter {
            // PUBLIC METHODS /////////////////////////////////////////////////
public:

            /// Constructor
            FieldFilterNone(void);

            /// Destructor
            ~FieldFilterNone(void);

            /** Create DM associated with filtered field.
             *
             * @param[in] dm PETSc DM of unfiltered field.
             * @param[in] description Description of input field.
             * @param[in] discretization Discretization of input field.
             */
            PetscDM createDM(PetscDM dm,
                             const pylith::topology::FieldBase::Description& description,
                             const pylith::topology::FieldBase::Discretization& discretization) const;

            /** Apply filter to global PETSc vector.
             *
             * @param[out] vectorOut Filtered global PETSc vector.
             * @param[in] dmOut PETSc DM associated with filtered global PETSc vector.
             * @param[in] vectorIn PETSc global vector to filter.
             */
            void apply(PetscVec* vectorOut,
                       PetscDM dmOut,
                       PetscVec vectorIn) const;

        }; // FieldFilterNone

    } // meshio
} // pylith

// End of file
