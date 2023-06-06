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
 * @file modulesrc/meshio/MeshIOLagrit.i
 *
 * @brief Python interface to C++ MeshIOLagrit object.
 */

namespace pylith {
    namespace meshio {
        class MeshIOLagrit: public MeshIO
        { // MeshIOLagrit
          // PUBLIC METHODS /////////////////////////////////////////////////
public:

            /// Constructor
            MeshIOLagrit(void);

            /// Destructor
            ~MeshIOLagrit(void);

            /// Deallocate PETSc and local data structures.
            void deallocate(void);

            /** Set filename for mesh GMV file.
             *
             * @param filename Name of file
             */
            void setFilenameGmv(const char* name);

            /** Get filename of mesh GMV file.
             *
             * @returns Name of file
             */
            const char* getFilenameGmv(void) const;

            /** Set filename for PSET mesh file.
             *
             * @param filename Name of file
             */
            void setFilenamePset(const char* name);

            /** Get filename of PSET mesh file.
             *
             * @returns Name of file
             */
            const char* getFilenamePset(void) const;

            /** Set flag to write ASCII or binary files.
             *
             * @param flag True if writing ASCII, false if writing binary
             */
            void setAsciiFlag(const bool flag);

            /** Get flag for writing ASCII or binary files.
             *
             * @returns True if writing ASCII, false if writing binary.
             */
            bool getAsciiFlag(void) const;

            /** Set flag to flip endian type when reading/writing from binary files.
             *
             * @param flag True if flipping endian, false otherwise
             */
            void setFlipEndian(const bool flag);

            /** Get flag for flipping endian type when reading/writing from
             * binary files.
             *
             * @returns True if flipping endian, false othewise.
             */
            bool getFlipEndian(void) const;

            /** Set flag indicating LaGriT Pset files use 32-bit integers.
             *
             * @param flag True if using 32-bit integers, false if using
             * 64-bit integers.
             */
            void setIOInt32(const bool flag);

            /** Get flag indicating LaGriT Pset files use 32-bit integers.
             *
             * @returns True if using 32-bit integers, false if using 64-bit integers.
             */
            bool getIOInt32(void) const;

            /** Set Fortran record header size flag.
             *
             * @param flag True if Fortran record header size is 32-bit,
             * false if 64-bit.
             */
            void setIsRecordHeader32Bit(const bool flag);

            /** Get Fortran record header size flag.
             *
             * @param returns True if Fortran record header size is 32-bit,
             *   false if 64-bit.
             */
            bool getIsRecordHeader32Bit(void) const;

            // PROTECTED METHODS //////////////////////////////////////////////
protected:

            /// Write mesh
            void _write(void) const;

            /// Read mesh
            void _read(void);

        }; // MeshIOLagrit

    } // meshio
} // pylith

// End of file
