// -*- C++ -*-
//
// ======================================================================
//
//                           Brad T. Aagaard
//                        U.S. Geological Survey
//
// {LicenseText}
//
// ======================================================================
//

/**
 * @file pylith/feassemble/Quadrature1D.hh
 *
 * @brief Quadrature for 1-D finite-elements.
 */

#if !defined(pylith_feassemble_quadrature1d_hh)
#define pylith_feassemble_quadrature1d_hh

#include "Quadrature.hh"

namespace pylith {
  namespace feassemble {
    class Quadrature1D;
    class TestQuadrature1D;
  } // feassemble
} // pylith

class pylith::feassemble::Quadrature1D : public Quadrature
{ // Quadrature1D
  friend class TestQuadrature1D; // unit testing

// PUBLIC MEMBERS ///////////////////////////////////////////////////////
public :

  /// Constructor
  Quadrature1D(void);

  /// Destructor
  ~Quadrature1D(void);

  /// Create a copy of this object.
  Quadrature* clone(void) const;

  void computeGeometry(const ALE::Obj<real_section_type>& coordinates,
		       const topology_type::point_type& cell);

// PROTECTED METHODS ////////////////////////////////////////////////////
protected :

  /** Copy constructor.
   *
   * @param q Quadrature to copy
   */
  Quadrature1D(const Quadrature1D& q);

// PRIVATE METHODS //////////////////////////////////////////////////////
private :

  const Quadrature1D& operator=(const Quadrature1D&); ///< Not implemented

}; // Quadrature1D

#include "Quadrature1D.icc" // inline methods

#endif // pylith_feassemble_quadrature1d_hh

// End of file 
