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
// Copyright (c) 2010-2016 University of California, Davis
//
// See COPYING for license information.
//
// ======================================================================
//

#include <portinfo>

#include "OutputSubfield.hh" // Implementation of class methods

#include "pylith/topology/Field.hh" // USES Field
#include "pylith/topology/FieldOps.hh" // USES FieldOps
#include "pylith/topology/Mesh.hh" // USES Mesh
#include "pylith/meshio/FieldFilter.hh" // USES FieldFilter

#include "pylith/utils/error.hh" // USES PYLITH_CHECK_ERROR

#include <typeinfo> // USES typeid()

// ------------------------------------------------------------------------------------------------
// Constructor
pylith::meshio::OutputSubfield::OutputSubfield(void) :
    _dm(NULL),
    _is(NULL),
    _vector(NULL),
    _filter(NULL),
    _filteredDM(NULL),
    _filteredVector(NULL) {}


// ------------------------------------------------------------------------------------------------
// Destructor
pylith::meshio::OutputSubfield::~OutputSubfield(void) {
    deallocate();
} // destructor


// ------------------------------------------------------------------------------------------------
// Deallocate PETSc and local data structures.
void
pylith::meshio::OutputSubfield::deallocate(void) {
    PetscErrorCode err;
    err = DMDestroy(&_dm);PYLITH_CHECK_ERROR(err);
    err = ISDestroy(&_is);PYLITH_CHECK_ERROR(err);
    err = VecDestroy(&_vector);PYLITH_CHECK_ERROR(err);

    err = DMDestroy(&_filteredDM);PYLITH_CHECK_ERROR(err);
    err = VecDestroy(&_filteredVector);PYLITH_CHECK_ERROR(err);

    _filter = NULL; // :TODO: Use shared pointer.
} // deallocate


// ------------------------------------------------------------------------------------------------
// Create OutputSubfield from Field.
pylith::meshio::OutputSubfield*
pylith::meshio::OutputSubfield::create(const pylith::topology::Field& field,
                                       const char* name,
                                       const pylith::meshio::FieldFilter* filter,
                                       const pylith::topology::Mesh* submesh) {
    PYLITH_METHOD_BEGIN;

    const pylith::topology::Field::SubfieldInfo& info = field.subfieldInfo(name);
    pylith::topology::Field::Discretization discretizationSubfield = info.fe;

    PetscDM subfieldDM = NULL;
    PetscIS subfieldIS = NULL;
    PetscErrorCode err;
    if (!submesh) {
        PetscBool hasConstraints = PETSC_FALSE;
        PetscBool ghasConstraints = PETSC_FALSE;
        err = PetscSectionHasConstraints(field.localSection(), &hasConstraints);PYLITH_CHECK_ERROR(err);
        MPI_Allreduce(&hasConstraints, &ghasConstraints, 1, MPIU_BOOL, MPI_LOR, field.mesh().comm());
        if (!ghasConstraints) {
            err = DMCreateSubDM(field.dmMesh(), 1, &info.index, &subfieldIS, &subfieldDM);PYLITH_CHECK_ERROR(err);
        } else {
            subfieldDM = pylith::topology::FieldOps::createSubfieldDM(field.dmMesh(), info.index);
            subfieldIS = pylith::topology::FieldOps::createSubfieldIS(field, name);
        } // if/else
    } else {
        assert(submesh);
        subfieldDM = pylith::topology::FieldOps::createSubdofDM(field.dmMesh(), submesh->dmMesh(), info.index);
        subfieldIS = pylith::topology::FieldOps::createSubdofIS(field, name, *submesh);

        // :KLUDGE: Modify discretization to be lower dimension.
        // We should really project the solution onto the lower dimension mesh to have the
        // proper discretization.
        discretizationSubfield.dimension = submesh->dimension();
        PetscFE fe = pylith::topology::FieldOps::createFE(discretizationSubfield, subfieldDM, info.description.numComponents);assert(fe);
        err = PetscFESetName(fe, info.description.label.c_str());PYLITH_CHECK_ERROR(err);
        err = DMSetField(subfieldDM, 0, NULL, (PetscObject)fe);PYLITH_CHECK_ERROR(err);
        err = DMSetFieldAvoidTensor(subfieldDM, 0, PETSC_TRUE);PYLITH_CHECK_ERROR(err);
        err = PetscFEDestroy(&fe);PYLITH_CHECK_ERROR(err);
        err = DMCreateDS(subfieldDM);PYLITH_CHECK_ERROR(err);
    } // if/else
    assert(subfieldDM);
    assert(subfieldIS);
    err = PetscObjectSetName((PetscObject)subfieldDM, name);PYLITH_CHECK_ERROR(err);

    OutputSubfield* subfield = new OutputSubfield(info.description, discretizationSubfield, subfieldDM, subfieldIS,
                                                  filter);assert(subfield);
    PYLITH_METHOD_RETURN(subfield);
}


// ------------------------------------------------------------------------------------------------
// Get description of subfield.
const pylith::topology::FieldBase::Description&
pylith::meshio::OutputSubfield::getDescription(void) const {
    return _description;
}


// ------------------------------------------------------------------------------------------------
// Get filtered PETSc global vector.
PetscVec
pylith::meshio::OutputSubfield::getVector(void) const {
    return _filteredVector;
}


// ------------------------------------------------------------------------------------------------
// Get PETSc DM for filtered vector.
PetscDM
pylith::meshio::OutputSubfield::getDM(void) const {
    return _filteredDM;
}


// ------------------------------------------------------------------------------------------------
// Extract subfield data from global PETSc vector with subfields.
void
pylith::meshio::OutputSubfield::extract(const PetscVec& fieldVector) {
    PYLITH_METHOD_BEGIN;
    assert(_is);
    assert(_vector);
    PetscErrorCode err = VecISCopy(fieldVector, _is, SCATTER_REVERSE, _vector);PYLITH_CHECK_ERROR(err);
    if (_filter) {
        assert(_filteredDM);
        _filter->apply(&_filteredVector, _filteredDM, _vector);
    } // if
    assert(_filteredVector);
    err = VecScale(_filteredVector, _description.scale);PYLITH_CHECK_ERROR(err);

    PYLITH_METHOD_END;
}


// ------------------------------------------------------------------------------------------------
// Constructor.
pylith::meshio::OutputSubfield::OutputSubfield(const pylith::topology::FieldBase::Description& description,
                                               const pylith::topology::FieldBase::Discretization& discretization,
                                               PetscDM dm,
                                               PetscIS is,
                                               const pylith::meshio::FieldFilter* filter) :
    _description(description),
    _discretization(discretization),
    _dm(dm),
    _is(is),
    _vector(NULL),
    _filter(filter),
    _filteredDM(NULL),
    _filteredVector(NULL) {
    PYLITH_METHOD_BEGIN;

    PetscErrorCode err;

    const char* const name = description.label.c_str();
    err = DMCreateGlobalVector(_dm, &_vector);PYLITH_CHECK_ERROR(err);
    err = PetscObjectSetName((PetscObject) _vector, name);PYLITH_CHECK_ERROR(err);
    if (_filter) {
        _filteredDM = _filter->createDM(_dm, description, discretization);
        err = PetscObjectSetName((PetscObject) _filteredDM, name);PYLITH_CHECK_ERROR(err);
    } else {
        _filteredDM = _dm;
        err = PetscObjectReference((PetscObject)_dm);PYLITH_CHECK_ERROR(err);
    } // if/else

    if (_filteredDM != _dm) {
        err = DMCreateGlobalVector(_filteredDM, &_filteredVector);PYLITH_CHECK_ERROR(err);
        err = PetscObjectSetName((PetscObject) _filteredVector, name);PYLITH_CHECK_ERROR(err);
    } else {
        _filteredVector = _vector;
        err = PetscObjectReference((PetscObject)_vector);PYLITH_CHECK_ERROR(err);
    } // if/else

    PYLITH_METHOD_END;
}


// End of file
