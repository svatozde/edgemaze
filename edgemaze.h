/* Generated by Cython 0.29.14 */

#ifndef __PYX_HAVE__edgemaze
#define __PYX_HAVE__edgemaze

#include "Python.h"

#ifndef __PYX_HAVE_API__edgemaze

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C void set_distance(PyObject *, __Pyx_memviewslice);
__PYX_EXTERN_C void set_direction(PyObject *, __Pyx_memviewslice);

#endif /* !__PYX_HAVE_API__edgemaze */

/* WARNING: the interface of the module init function changed in CPython 3.5. */
/* It now returns a PyModuleDef instance instead of a PyModule instance. */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initedgemaze(void);
#else
PyMODINIT_FUNC PyInit_edgemaze(void);
#endif

#endif /* !__PYX_HAVE__edgemaze */
