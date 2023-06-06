// -*- C++ -*-
//
// ======================================================================
//
// Brad T. Aagaard, U.S. Geological Survey
// Charles A. Williams, GNS Science
// Matthew G. Knepley, University at Buffalo
//
// This code was developed as part of the Computational Infrastructure
// for Geodynamics (http://geodynamics.org).
//
// Copyright (c) 2010-2022 University of California, Davis
//
// See LICENSE.md for license information.
//
// ======================================================================
//

/**
 * @file libsrc/meshio/OutputSolnBoundary.hh
 *
 * @brief C++ object for managing solution output over a boundary.
 */

#if !defined(pylith_meshio_outputsolnboundary_hh)
#define pylith_meshio_outputsolnboundary_hh

#include "meshiofwd.hh" // forward declarations

#include "pylith/meshio/OutputSoln.hh" // ISA OutputSoln

#include "pylith/topology/topologyfwd.hh" // HOLDSA Mesh

#include <string> // HASA std::string

class pylith::meshio::OutputSolnBoundary : public pylith::meshio::OutputSoln {
    friend class TestOutputSolnBoundary; // unit testing

    // PUBLIC METHODS //////////////////////////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor.
    OutputSolnBoundary(void);

    /// Destructor
    ~OutputSolnBoundary(void);

    /// Deallocate PETSc and local data structures.
    void deallocate(void);

    /** Set name of label identifier for subdomain.
     *
     * @param[in] value Name of label for subdomain.
     */
    void setLabelName(const char* value);

    /** Set value of label identifier for subdomain.
     *
     * @param[in] value Value of label for subdomain.
     */
    void setLabelValue(const int value);

    /** Verify configuration.
     *
     * @param[in] solution Solution field.
     */
    void verifyConfiguration(const pylith::topology::Field& solution) const;

    // PROTECTED METHODS ///////////////////////////////////////////////////////////////////////////////////////////////
protected:

    /** Write solution at time step.
     *
     * @param[in] t Current time.
     * @param[in] tindex Current time step.
     * @param[in] solution Solution at time t.
     */
    void _writeSolnStep(const PylithReal t,
                        const PylithInt tindex,
                        const pylith::topology::Field& solution);

    // PRIVATE MEMBERS /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    pylith::topology::Mesh* _boundaryMesh; ///< Mesh of subdomain.
    std::string _labelName; ///< Name of label for subdomain.
    int _labelValue; ///< Value of label for subdomain.

    // NOT IMPLEMENTED /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    OutputSolnBoundary(const OutputSolnBoundary&); ///< Not implemented.
    const OutputSolnBoundary& operator=(const OutputSolnBoundary&); ///< Not implemented

}; // OutputSolnBoundary

#endif // pylith_meshio_outputsolnboundary_hh

// End of file
