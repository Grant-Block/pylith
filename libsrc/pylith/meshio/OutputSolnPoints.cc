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

#include <portinfo>

#include "OutputSolnPoints.hh" // implementation of class methods

#include "pylith/meshio/DataWriter.hh" // USES DataWriter
#include "pylith/meshio/MeshBuilder.hh" // USES MeshBuilder

#include "pylith/topology/Mesh.hh" // USES Mesh
#include "pylith/meshio/OutputSubfield.hh" // USES OutputSubfield
#include "pylith/topology/MeshOps.hh" // USES MeshOps::nondimensionalize()
#include "pylith/topology/Stratum.hh" // USES Stratum
#include "pylith/topology/VisitorMesh.hh" // USES VecVisitorMesh

#include "pylith/utils/journals.hh" // USES PYLITH_COMPONENT_*

#include "spatialdata/geocoords/CoordSys.hh" // USES CoordSys
#include "spatialdata/geocoords/Converter.hh" // USES Converter
#include "spatialdata/units/Nondimensional.hh" // USES Nondimensional

#include <cassert> // USES assert()

// ---------------------------------------------------------------------------------------------------------------------
// Constructor
pylith::meshio::OutputSolnPoints::OutputSolnPoints(void) :
    _pointsMesh(NULL),
    _pointsSoln(NULL),
    _interpolator(NULL) {
    PyreComponent::setName("outputsolnpoints");
} // constructor


// ---------------------------------------------------------------------------------------------------------------------
// Destructor
pylith::meshio::OutputSolnPoints::~OutputSolnPoints(void) {
    deallocate();
} // destructor


// ---------------------------------------------------------------------------------------------------------------------
// Deallocate PETSc and local data structures.
void
pylith::meshio::OutputSolnPoints::deallocate(void) {
    PYLITH_METHOD_BEGIN;

    OutputSoln::deallocate();

    PetscErrorCode err = DMInterpolationDestroy(&_interpolator);PYLITH_CHECK_ERROR(err);

    delete _pointsMesh;_pointsMesh = NULL;
    delete _pointsSoln;_pointsSoln = NULL;

    PYLITH_METHOD_END;
} // deallocate


// ---------------------------------------------------------------------------------------------------------------------
// Set point names and coordinates of points .
void
pylith::meshio::OutputSolnPoints::setPoints(const PylithReal* pointCoords,
                                            const PylithInt numPoints,
                                            const PylithInt spaceDim,
                                            const char* const* pointNames,
                                            const PylithInt numPointNames) {
    PYLITH_METHOD_BEGIN;

    assert(pointCoords && pointNames);
    assert(numPoints == numPointNames);

    // Copy point coordinates.
    const PylithInt size = numPoints * spaceDim;
    _pointCoords.resize(size);
    for (PylithInt i = 0; i < size; ++i) {
        _pointCoords[i] = pointCoords[i];
    } // for

    // Copy point names.
    _pointNames.resize(numPointNames);
    for (PylithInt i = 0; i < numPointNames; ++i) {
        _pointNames[i] = pointNames[i];
    } // for

    PYLITH_METHOD_END;
} // setPoints


// ---------------------------------------------------------------------------------------------------------------------
// Write solution at time step.
void
pylith::meshio::OutputSolnPoints::_writeSolnStep(const PylithReal t,
                                                 const PylithInt tindex,
                                                 const pylith::topology::Field& solution) {
    PYLITH_METHOD_BEGIN;
    PYLITH_COMPONENT_DEBUG("_writeSolnStep(t="<<t<<", tindex="<<tindex<<", solution="<<solution.getLabel()<<")");

    if (!_interpolator) {
        _setupInterpolator(solution);
    } // if
    _interpolateField(solution);
    assert(_pointsSoln);
    _pointsSoln->scatterLocalToOutput();
    const PetscVec solutionVector = _pointsSoln->outputVector();assert(solutionVector);

    assert(_pointsMesh);
    const bool writePointNames = !_writer->isOpen();
    _openSolnStep(t, *_pointsMesh);
    if (writePointNames) { _writePointNames(); }

    const pylith::string_vector& subfieldNames = _expandSubfieldNames(solution);
    const size_t numSubfieldNames = subfieldNames.size();
    for (size_t iField = 0; iField < numSubfieldNames; iField++) {
        assert(solution.hasSubfield(subfieldNames[iField].c_str()));

        OutputSubfield* subfield = NULL;
        subfield = OutputObserver::_getSubfield(*_pointsSoln, subfieldNames[iField].c_str(), _pointsMesh);assert(subfield);
        subfield->extract(solutionVector);

        OutputObserver::_appendField(t, *subfield);
    } // for
    _closeSolnStep();

    PYLITH_METHOD_END;
} // _writeDataSet


// ---------------------------------------------------------------------------------------------------------------------
// Setup interpolator.
void
pylith::meshio::OutputSolnPoints::_setupInterpolator(const pylith::topology::Field& solution) {
    PYLITH_METHOD_BEGIN;

    PetscErrorCode err = DMInterpolationDestroy(&_interpolator);PYLITH_CHECK_ERROR(err);
    assert(!_interpolator);

    const spatialdata::geocoords::CoordSys* csMesh = solution.mesh().getCoordSys();assert(csMesh);
    const int spaceDim = csMesh->getSpaceDim();

    MPI_Comm comm = solution.mesh().comm();

    // Setup interpolator object
    PetscDM dmSoln = solution.dmMesh();assert(dmSoln);

    err = DMInterpolationCreate(comm, &_interpolator);PYLITH_CHECK_ERROR(err);
    err = DMInterpolationSetDim(_interpolator, spaceDim);PYLITH_CHECK_ERROR(err);
    err = DMInterpolationAddPoints(_interpolator, _pointCoords.size(), (PetscReal*) &_pointCoords[0]);PYLITH_CHECK_ERROR(err);
    const PetscBool pointsAllProcs = PETSC_TRUE;
    const PetscBool ignoreOutsideDomain = PETSC_FALSE;
    err = DMInterpolationSetUp(_interpolator, dmSoln, pointsAllProcs, ignoreOutsideDomain);PYLITH_CHECK_ERROR(err);

    // Create mesh corresponding to local points.
    const int meshDim = 0;
    delete _pointsMesh;_pointsMesh = new pylith::topology::Mesh(meshDim, comm);assert(_pointsMesh);

    const int numVertices = _interpolator->n; // Number of local points
    PylithScalar* pointCoordsLocal = NULL;
    err = VecGetArray(_interpolator->coords, &pointCoordsLocal);PYLITH_CHECK_ERROR(err);
    const int sizeLocal = numVertices * spaceDim;
    scalar_array vertices(sizeLocal);
    for (PylithInt i = 0; i < sizeLocal; ++i) {
        vertices[i] = pointCoordsLocal[i];
    } // for
    const int numCells = numVertices;
    const int numCorners = 1;
    pylith::int_array cells(numCells * numCorners);
    for (PylithInt i = 0; i < numCells; ++i) {
        cells[i] = i;
    } // for
    const bool isParallel = true;
    MeshBuilder::buildMesh(_pointsMesh, &vertices, numVertices, spaceDim,
                           cells, numCells, numCorners, meshDim, isParallel);
    err = VecRestoreArray(_interpolator->coords, &pointCoordsLocal);PYLITH_CHECK_ERROR(err);

    // Set coordinate system and create nondimensionalized coordinates
    _pointsMesh->setCoordSys(csMesh);

    PylithReal lengthScale = 1.0;
    err = DMPlexGetScale(dmSoln, PETSC_UNIT_LENGTH, &lengthScale);PYLITH_CHECK_ERROR(err);
    err = DMPlexSetScale(_pointsMesh->dmMesh(), PETSC_UNIT_LENGTH, lengthScale);PYLITH_CHECK_ERROR(err);

#if 0 // DEBUGGING
    _pointsMesh->view("::ascii_info_detail");
#endif

    // Upate point names to only local points.
    pylith::string_vector pointNamesLocal(numVertices);
    const size_t numPoints = _pointNames.size();
    for (PylithInt iVertex = 0; iVertex < numVertices; ++iVertex) {
        // Find point in array of all points to get index for point name.
        for (size_t ipoint = 0; ipoint < numPoints; ++ipoint) {
            const PylithReal tolerance = 1.0e-6;
            PylithReal dist = 0.0;
            for (int iDim = 0; iDim < spaceDim; ++iDim) {
                dist += pow(_pointCoords[ipoint*spaceDim+iDim] - vertices[iVertex*spaceDim+iDim], 2);
            } // for
            if (sqrt(dist) < tolerance) {
                pointNamesLocal[iVertex] = _pointNames[ipoint];
                break;
            } // if
        } // for
    } // for

    _pointNames = pointNamesLocal;
    _pointCoords.resize(0);

    // Determine size of interpolated field that we will have.
    PetscInt numDof = 0;
    const pylith::string_vector& subfieldNames = solution.subfieldNames();
    const size_t numSubfields = subfieldNames.size();
    for (size_t i = 0; i < numSubfields; ++i) {
        numDof += solution.subfieldInfo(subfieldNames[i].c_str()).description.numComponents;
    } // for
    err = DMInterpolationSetDof(_interpolator, numDof);PYLITH_CHECK_ERROR(err);

    const PetscDM dmPoints = _pointsMesh->dmMesh();assert(dmPoints);
    pylith::topology::Stratum pointsStratum(dmPoints, pylith::topology::Stratum::DEPTH, 0);

    delete _pointsSoln;_pointsSoln = new pylith::topology::Field(*_pointsMesh);
    for (size_t i = 0; i < subfieldNames.size(); ++i) {
        const pylith::topology::Field::SubfieldInfo& sinfo = solution.subfieldInfo(subfieldNames[i].c_str());
        _pointsSoln->subfieldAdd(sinfo.description, sinfo.fe);
    } // for
    _pointsSoln->subfieldsSetup();
    _pointsSoln->createDiscretization();
    _pointsSoln->setLabel(solution.getLabel());
    _pointsSoln->allocate();
    _pointsSoln->zeroLocal();
    _pointsSoln->createOutputVector();

    PYLITH_METHOD_END;
} // setupInterpolator


// ---------------------------------------------------------------------------------------------------------------------
// Append finite-element field to file.
void
pylith::meshio::OutputSolnPoints::_interpolateField(const pylith::topology::Field& solution) {
    PYLITH_METHOD_BEGIN;

    PetscErrorCode err;
    PetscVec interpolatedLocalVec = NULL;
    err = DMInterpolationGetVector(_interpolator, &interpolatedLocalVec);PYLITH_CHECK_ERROR(err);
    err = DMInterpolationEvaluate(_interpolator, solution.dmMesh(), solution.localVector(), interpolatedLocalVec);PYLITH_CHECK_ERROR(err);

    PYLITH_METHOD_END;
} // appendVertexField


// ---------------------------------------------------------------------------------------------------------------------
// Write dataset with names of points to file.
void
pylith::meshio::OutputSolnPoints::_writePointNames(void) {
    PYLITH_METHOD_BEGIN;

    assert(_writer);
    _writer->writePointNames(_pointNames, *_pointsMesh);

    PYLITH_METHOD_END;
} // writePointNames


// End of file
