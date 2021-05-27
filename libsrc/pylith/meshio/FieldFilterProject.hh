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
 * @file libsrc/meshio/FieldFilterProject.hh
 *
 * @brief C++ object for projecting a field to another basis.
 */

#if !defined(pylith_meshio_fieldfilterproject_hh)
#define pylith_meshio_fieldfilterproject_hh

#include "FieldFilter.hh" // ISA FieldFilter

class pylith::meshio::FieldFilterProject : public FieldFilter {
    // PUBLIC METHODS //////////////////////////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor
    FieldFilterProject(void);

    /// Destructor
    ~FieldFilterProject(void);

    /** Set basis order for projected field.
     *
     * @param[in] value Basis order.
     */
    void setBasisOrder(const int value);

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

    // PRIVATE MEMBERS /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    PetscPointFunc _passThruFns[1]; ///< Pass through point functions.
    int _basisOrder; ///< Basis order for projected field.

    // NOT IMPLEMENTED /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    FieldFilterProject(const FieldFilterProject& f); ///< Not implemented.
    const FieldFilterProject& operator=(const FieldFilterProject&); ///< Not implemented.

}; // FieldFilterProject

#endif // pylith_meshio_fieldfilterproject_hh

// End of file
