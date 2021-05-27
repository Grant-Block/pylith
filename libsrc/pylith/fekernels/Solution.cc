/* -*- C++ -*-
 *
 * ----------------------------------------------------------------------
 *
 * Brad T. Aagaard, U.S. Geological Survey
 * Charles A. Williams, GNS Science
 * Matthew G. Knepley, University of Chicago
 *
 * This code was developed as part of the Computational Infrastructure
 * for Geodynamics (http:*geodynamics.org).
 *
 * Copyright (c) 2010-2015 University of California, Davis
 *
 * See COPYING for license information.
 *
 * ----------------------------------------------------------------------
 */

#include <portinfo>

#include "pylith/fekernels/Solution.hh"

#include <cassert> // USES assert()

// ---------------------------------------------------------------------------------------------------------------------
// Identify function kernel.
void
pylith::fekernels::Solution::passThruSubfield(const PylithInt dim,
                                              const PylithInt numS,
                                              const PylithInt numA,
                                              const PylithInt sOff[],
                                              const PylithInt sOff_x[],
                                              const PylithScalar s[],
                                              const PylithScalar s_t[],
                                              const PylithScalar s_x[],
                                              const PylithInt aOff[],
                                              const PylithInt aOff_x[],
                                              const PylithScalar a[],
                                              const PylithScalar a_t[],
                                              const PylithScalar a_x[],
                                              const PylithReal t,
                                              const PylithScalar x[],
                                              const PylithInt numConstants,
                                              const PylithScalar constants[],
                                              PylithScalar field[]) {
    assert(s);
    assert(sOff);
    assert(field);
    assert(1 == numS);

    const PylithInt sEnd = sOff[1];
    for (PylithInt i = sOff[0]; i < sEnd; ++i) {
        if (isnan(s[i])) {
            int i = 0;
        }
        field[i] = s[i];
    } // for
} // passThruSubfield


// End of file
