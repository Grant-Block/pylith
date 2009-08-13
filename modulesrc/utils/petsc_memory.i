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

// ----------------------------------------------------------------------
namespace ALE {
  class MemoryLogger {
  protected:
    MemoryLogger();
  public:
    /** Return the global logger
     *
     * @returns the logger
     */
    static MemoryLogger& singleton();
    int debug();
    void setDebug(int debug);
  public:
    void stagePush(const char *name);
    void stagePop();
    void logAllocation(const char *className, int bytes);
    void logAllocation(const char *stage, const char *className, int bytes);
    void logDeallocation(const char *className, int bytes);
    void logDeallocation(const char *stage, const char *className, int bytes);
  public:
    int getNumAllocations();
    int getNumAllocations(const char *stage);
    int getNumDeallocations();
    int getNumDeallocations(const char *stage);
    int getAllocationTotal();
    int getAllocationTotal(const char *stage);
    int getDeallocationTotal();
    int getDeallocationTotal(const char *stage);
    void show();
  };
}

// End of file
