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

#include "MeshOps.hh" // implementation of class methods

#include "pylith/topology/Mesh.hh" // USES Mesh
#include "pylith/topology/Stratum.hh" // USES Stratum
#include "pylith/utils/array.hh" // USES int_array

#include "spatialdata/units/Nondimensional.hh" // USES Nondimensional

#include <stdexcept> // USES std::runtime_error
#include <sstream> // USES std::ostringstream
#include <cassert> // USES assert()

#include <algorithm> // USES std::sort, std::find
#include <map> // USES std::map

// ---------------------------------------------------------------------------------------------------------------------
// Create subdomain mesh using label.
pylith::topology::Mesh*
pylith::topology::MeshOps::createSubdomainMesh(const pylith::topology::Mesh& mesh,
                                               const char* label,
                                               const int labelValue,
                                               const char* descriptiveLabel) {
    PYLITH_METHOD_BEGIN;

    assert(label);

    PetscDM dmDomain = mesh.dmMesh();assert(dmDomain);
    PetscErrorCode err = 0;

    PetscBool hasLabel = PETSC_FALSE;
    err = DMHasLabel(dmDomain, label, &hasLabel);PYLITH_CHECK_ERROR(err);
    if (!hasLabel) {
        std::ostringstream msg;
        msg << "Could not find group of points '" << label << "' in PETSc DM mesh.";
        throw std::runtime_error(msg.str());
    } // if

    /* TODO: Add creation of pointSF for submesh */
    PetscDMLabel dmLabel = NULL;
    err = DMGetLabel(dmDomain, label, &dmLabel);PYLITH_CHECK_ERROR(err);

    PetscBool hasLabelValue = PETSC_FALSE;
    err = DMLabelHasValue(dmLabel, labelValue, &hasLabelValue);PYLITH_CHECK_ERROR(err);
    if (!hasLabelValue) {
        std::ostringstream msg;
        msg << "Could not find value " << labelValue << " in label '" << label << "' in PETSc DM mesh.";
        throw std::runtime_error(msg.str());
    } // if

    PetscDM dmSubdomain = NULL;
    err = DMPlexFilter(dmDomain, dmLabel, labelValue, &dmSubdomain);PYLITH_CHECK_ERROR(err);

    PetscInt maxConeSizeLocal = 0, maxConeSize = 0;
    err = DMPlexGetMaxSizes(dmSubdomain, &maxConeSizeLocal, NULL);PYLITH_CHECK_ERROR(err);
    err = MPI_Allreduce(&maxConeSizeLocal, &maxConeSize, 1, MPI_INT, MPI_MAX,
                        PetscObjectComm((PetscObject) dmSubdomain));PYLITH_CHECK_ERROR(err);

    if (maxConeSize <= 0) {
        err = DMDestroy(&dmSubdomain);PYLITH_CHECK_ERROR(err);
        std::ostringstream msg;
        msg << "Error while creating mesh of subdomain. Subdomain mesh '" << label << "' does not contain any cells.\n";
        throw std::runtime_error(msg.str());
    } // if

    // Set name
    err = PetscObjectSetName((PetscObject) dmSubdomain, descriptiveLabel);PYLITH_CHECK_ERROR(err);

    // Set lengthscale
    PylithScalar lengthScale;
    err = DMPlexGetScale(dmDomain, PETSC_UNIT_LENGTH, &lengthScale);PYLITH_CHECK_ERROR(err);
    err = DMPlexSetScale(dmSubdomain, PETSC_UNIT_LENGTH, lengthScale);PYLITH_CHECK_ERROR(err);

    pylith::topology::Mesh* submesh = new pylith::topology::Mesh(true);assert(submesh);
    submesh->setCoordSys(mesh.getCoordSys());
    submesh->dmMesh(dmSubdomain);

    PYLITH_METHOD_RETURN(submesh);
} // createSubdomainMesh


// ---------------------------------------------------------------------------------------------------------------------
// Create lower dimension mesh using label.
pylith::topology::Mesh*
pylith::topology::MeshOps::createLowerDimMesh(const pylith::topology::Mesh& mesh,
                                              const char* label) {
    PYLITH_METHOD_BEGIN;

    assert(label);

    PetscDM dmDomain = mesh.dmMesh();assert(dmDomain);
    PetscErrorCode err = 0;

    PetscBool hasLabel = PETSC_FALSE;
    err = DMHasLabel(dmDomain, label, &hasLabel);PYLITH_CHECK_ERROR(err);
    if (!hasLabel) {
        std::ostringstream msg;
        msg << "Could not find group of points '" << label << "' in PETSc DM mesh.";
        throw std::runtime_error(msg.str());
    } // if

    if (mesh.dimension() < 1) {
        throw std::logic_error("INTERNAL ERROR in MeshOps::createLowerDimMesh()\n"
                               "Cannot create submesh for mesh with dimension < 1.");
    } // if

    /* TODO: Add creation of pointSF for submesh */
    PetscDM dmSubmesh = NULL;
    PetscDMLabel dmLabel = NULL;
    err = DMGetLabel(dmDomain, label, &dmLabel);PYLITH_CHECK_ERROR(err);
    PetscBool hasLabelValue = PETSC_FALSE;
    err = DMLabelHasValue(dmLabel, 1, &hasLabelValue);PYLITH_CHECK_ERROR(err);
    if (!hasLabelValue) {
        std::ostringstream msg;
        msg << "Could not find value 1 in label for group of points '" << label << "' in PETSc DM mesh.";
        throw std::logic_error(msg.str());
    } // if

    err = DMPlexCreateSubmesh(dmDomain, dmLabel, 1, PETSC_FALSE, &dmSubmesh);PYLITH_CHECK_ERROR(err);

    PetscInt maxConeSizeLocal = 0, maxConeSize = 0;
    err = DMPlexGetMaxSizes(dmSubmesh, &maxConeSizeLocal, NULL);PYLITH_CHECK_ERROR(err);
    err = MPI_Allreduce(&maxConeSizeLocal, &maxConeSize, 1, MPI_INT, MPI_MAX,
                        PetscObjectComm((PetscObject) dmSubmesh));PYLITH_CHECK_ERROR(err);

    if (maxConeSize <= 0) {
        err = DMDestroy(&dmSubmesh);PYLITH_CHECK_ERROR(err);
        std::ostringstream msg;
        msg << "Error while creating lower dimension mesh. Submesh '" << label << "' does not contain any cells.\n";
        throw std::runtime_error(msg.str());
    } // if

    // Set name
    std::string meshLabel = "subdomain_" + std::string(label);
    err = PetscObjectSetName((PetscObject) dmSubmesh, meshLabel.c_str());PYLITH_CHECK_ERROR(err);

    // Set lengthscale
    PylithScalar lengthScale;
    err = DMPlexGetScale(dmDomain, PETSC_UNIT_LENGTH, &lengthScale);PYLITH_CHECK_ERROR(err);
    err = DMPlexSetScale(dmSubmesh, PETSC_UNIT_LENGTH, lengthScale);PYLITH_CHECK_ERROR(err);

    pylith::topology::Mesh* submesh = new pylith::topology::Mesh(true);assert(submesh);
    submesh->setCoordSys(mesh.getCoordSys());
    submesh->dmMesh(dmSubmesh);

    // Check topology
    MeshOps::checkTopology(*submesh);

    PYLITH_METHOD_RETURN(submesh);
} // createLowerDimMesh


// ---------------------------------------------------------------------------------------------------------------------
// Nondimensionalize the finite-element mesh.
void
pylith::topology::MeshOps::nondimensionalize(Mesh* const mesh,
                                             const spatialdata::units::Nondimensional& normalizer) {
    PYLITH_METHOD_BEGIN;

    assert(mesh);

    PetscVec coordVec = NULL;
    const PylithScalar lengthScale = normalizer.getLengthScale();
    PetscErrorCode err = 0;

    PetscDM dmMesh = mesh->dmMesh();assert(dmMesh);
    err = DMGetCoordinatesLocal(dmMesh, &coordVec);PYLITH_CHECK_ERROR(err);assert(coordVec);
    err = VecScale(coordVec, 1.0/lengthScale);PYLITH_CHECK_ERROR(err);
    err = DMPlexSetScale(dmMesh, PETSC_UNIT_LENGTH, lengthScale);PYLITH_CHECK_ERROR(err);
    err = DMViewFromOptions(dmMesh, NULL, "-pylith_nondim_dm_view");PYLITH_CHECK_ERROR(err);

    const PetscInt dim = mesh->dimension();
    if (dim < 1) {
        PYLITH_METHOD_END;
    } // if
    PylithReal coordMin[3];
    PylithReal coordMax[3];
    err = DMGetBoundingBox(dmMesh, coordMin, coordMax);
    PylithReal volume = 1.0;
    for (int i = 0; i < dim; ++i) {
        volume *= coordMax[i] - coordMin[i];
    } // for
    assert(dim > 0);
    const PylithReal avgCellDim = pow(volume / mesh->numCells(), 1.0/dim);
    const PylithReal avgDimTolerance = 0.02;
    if (avgCellDim < avgDimTolerance) {
        std::ostringstream msg;
        msg << "Nondimensional average cell dimension (" << avgCellDim << ") is less than minimum tolerance ("
            << avgDimTolerance << "). This usually means the length scale (" << lengthScale << ") used in the "
            << "nondimensionalization needs to be smaller. Based on the average cell size, a value of about "
            << pow(10, int(log10(avgCellDim*lengthScale))) << " should be appropriate.";
        throw std::runtime_error(msg.str());
    } // if/else

    PYLITH_METHOD_END;
} // nondimensionalize


// ---------------------------------------------------------------------------------------------------------------------
bool
pylith::topology::MeshOps::isCohesiveCell(const PetscDM dmMesh,
                                          const PetscInt cell) {
    bool isCohesive = false;

    DMPolytopeType ct;
    PetscErrorCode err = DMPlexGetCellType(dmMesh, cell, &ct);PYLITH_CHECK_ERROR(err);
    if ((ct == DM_POLYTOPE_SEG_PRISM_TENSOR) ||
        (ct == DM_POLYTOPE_TRI_PRISM_TENSOR) ||
        (ct == DM_POLYTOPE_QUAD_PRISM_TENSOR)) { isCohesive = true; }

    return isCohesive;
} // isCohesiveCell


// ---------------------------------------------------------------------------------------------------------------------
// Check topology of mesh.
void
pylith::topology::MeshOps::checkTopology(const Mesh& mesh) {
    PetscDM dmMesh = mesh.dmMesh();assert(dmMesh);

    DMLabel subpointMap;
    PetscErrorCode ierr = DMPlexGetSubpointMap(dmMesh, &subpointMap);PYLITH_CHECK_ERROR(ierr);
    PetscInt cellHeight = subpointMap ? 1 : 0;

    PetscErrorCode err;
    err = DMViewFromOptions(dmMesh, NULL, "-pylith_checktopo_dm_view");PYLITH_CHECK_ERROR(err);
    err = DMPlexCheckSymmetry(dmMesh);PYLITH_CHECK_ERROR_MSG(err, "Error in topology of mesh associated with symmetry of adjacency information.");

    err = DMPlexCheckSkeleton(dmMesh, cellHeight);PYLITH_CHECK_ERROR_MSG(err, "Error in topology of mesh cells.");
} // checkTopology


// ---------------------------------------------------------------------------------------------------------------------
bool
pylith::topology::MeshOps::isSimplexMesh(const Mesh& mesh) {
    PYLITH_METHOD_BEGIN;

    bool isSimplex = false;

    const PetscDM dm = mesh.dmMesh();
    PetscInt closureSize, vStart, vEnd;
    PetscInt* closure = NULL;
    PetscErrorCode err;
    const int dim = mesh.dimension();
    err = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);PYLITH_CHECK_ERROR(err);
    err = DMPlexGetTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure);PYLITH_CHECK_ERROR(err);
    PetscInt numVertices = 0;
    for (PetscInt c = 0; c < closureSize*2; c += 2) {
        if ((closure[c] >= vStart) && (closure[c] < vEnd)) {
            ++numVertices;
        } // if
    } // for
    if (numVertices == dim+1) {
        isSimplex = PETSC_TRUE;
    } // if
    err = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure);PYLITH_CHECK_ERROR(err);

    PYLITH_METHOD_RETURN(isSimplex);
} // isSimplexMesh


// ---------------------------------------------------------------------------------------------------------------------
void
pylith::topology::MeshOps::checkMaterialIds(const pylith::topology::Mesh& mesh,
                                            pylith::int_array& materialIds) {
    PYLITH_METHOD_BEGIN;

    PetscErrorCode err;

    // Create map with indices for each material
    const size_t numIds = materialIds.size();
    std::map<int, int> materialIndex;
    for (size_t i = 0; i < numIds; ++i) {
        materialIndex[materialIds[i]] = i;
    } // for

    int_array matCellCounts(numIds);
    matCellCounts = 0;

    PetscDM dmMesh = mesh.dmMesh();assert(dmMesh);
    Stratum cellsStratum(dmMesh, Stratum::HEIGHT, 0);
    const PetscInt cStart = cellsStratum.begin();
    const PetscInt cEnd = cellsStratum.end();

    PetscDMLabel materialsLabel = NULL;
    const char* const labelName = pylith::topology::Mesh::getCellsLabelName();
    err = DMGetLabel(dmMesh, labelName, &materialsLabel);PYLITH_CHECK_ERROR(err);assert(materialsLabel);

    int *matBegin = &materialIds[0];
    int *matEnd = &materialIds[0] + materialIds.size();
    std::sort(matBegin, matEnd);

    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt matId;

        err = DMLabelGetValue(materialsLabel, c, &matId);PYLITH_CHECK_ERROR(err);
        if (matId < 0) {
            // :KLUDGE: Skip cells that are probably hybrid cells in halo
            // around fault that we currently ignore when looping over
            // materials (including cohesive cells).
            continue;
        } // if
        const int *result = std::find(matBegin, matEnd, matId);
        if (result == matEnd) {
            std::ostringstream msg;
            msg << "Material id '" << matId << "' for cell '" << c
                << "' does not match the id of any available materials or interfaces.";
            throw std::runtime_error(msg.str());
        } // if

        const size_t matIndex = materialIndex[matId];
        assert(0 <= matIndex && matIndex < numIds);
        ++matCellCounts[matIndex];
    } // for

    // Make sure each material has cells.
    int_array matCellCountsAll(matCellCounts.size());
    err = MPI_Allreduce(&matCellCounts[0], &matCellCountsAll[0],
                        matCellCounts.size(), MPI_INT, MPI_SUM, mesh.comm());PYLITH_CHECK_ERROR(err);
    for (size_t i = 0; i < numIds; ++i) {
        const int matId = materialIds[i];
        const size_t matIndex = materialIndex[matId];
        assert(0 <= matIndex && matIndex < numIds);
        if (matCellCountsAll[matIndex] <= 0) {
            std::ostringstream msg;
            msg << "No cells associated with material with id '" << matId << "'.";
            throw std::runtime_error(msg.str());
        } // if
    } // for

    PYLITH_METHOD_END;
} // checkMaterialIds


// End of file
