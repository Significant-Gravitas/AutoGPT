/*
 * Summary: macros for marking symbols as exportable/importable.
 * Description: macros for marking symbols as exportable/importable.
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Igor Zlatkovic <igor@zlatkovic.com>
 */

#ifndef __XSLT_EXPORTS_H__
#define __XSLT_EXPORTS_H__

/**
 * XSLTPUBFUN:
 * XSLTPUBFUN, XSLTPUBVAR, XSLTCALL
 *
 * Macros which declare an exportable function, an exportable variable and
 * the calling convention used for functions.
 *
 * Please use an extra block for every platform/compiler combination when
 * modifying this, rather than overlong #ifdef lines. This helps
 * readability as well as the fact that different compilers on the same
 * platform might need different definitions.
 */

/**
 * XSLTPUBFUN:
 *
 * Macros which declare an exportable function
 */
#define XSLTPUBFUN
/**
 * XSLTPUBVAR:
 *
 * Macros which declare an exportable variable
 */
#define XSLTPUBVAR extern
/**
 * XSLTCALL:
 *
 * Macros which declare the called convention for exported functions
 */
#define XSLTCALL

/** DOC_DISABLE */

/* Windows platform with MS compiler */
#if defined(_WIN32) && defined(_MSC_VER)
  #undef XSLTPUBFUN
  #undef XSLTPUBVAR
  #undef XSLTCALL
  #if defined(IN_LIBXSLT) && !defined(LIBXSLT_STATIC)
    #define XSLTPUBFUN __declspec(dllexport)
    #define XSLTPUBVAR __declspec(dllexport)
  #else
    #define XSLTPUBFUN
    #if !defined(LIBXSLT_STATIC)
      #define XSLTPUBVAR __declspec(dllimport) extern
    #else
      #define XSLTPUBVAR extern
    #endif
  #endif
  #define XSLTCALL __cdecl
  #if !defined _REENTRANT
    #define _REENTRANT
  #endif
#endif

/* Windows platform with Borland compiler */
#if defined(_WIN32) && defined(__BORLANDC__)
  #undef XSLTPUBFUN
  #undef XSLTPUBVAR
  #undef XSLTCALL
  #if defined(IN_LIBXSLT) && !defined(LIBXSLT_STATIC)
    #define XSLTPUBFUN __declspec(dllexport)
    #define XSLTPUBVAR __declspec(dllexport) extern
  #else
    #define XSLTPUBFUN
    #if !defined(LIBXSLT_STATIC)
      #define XSLTPUBVAR __declspec(dllimport) extern
    #else
      #define XSLTPUBVAR extern
    #endif
  #endif
  #define XSLTCALL __cdecl
  #if !defined _REENTRANT
    #define _REENTRANT
  #endif
#endif

/* Windows platform with GNU compiler (Mingw) */
#if defined(_WIN32) && defined(__MINGW32__)
  #undef XSLTPUBFUN
  #undef XSLTPUBVAR
  #undef XSLTCALL
/*
  #if defined(IN_LIBXSLT) && !defined(LIBXSLT_STATIC)
*/
  #if !defined(LIBXSLT_STATIC)
    #define XSLTPUBFUN __declspec(dllexport)
    #define XSLTPUBVAR __declspec(dllexport) extern
  #else
    #define XSLTPUBFUN
    #if !defined(LIBXSLT_STATIC)
      #define XSLTPUBVAR __declspec(dllimport) extern
    #else
      #define XSLTPUBVAR extern
    #endif
  #endif
  #define XSLTCALL __cdecl
  #if !defined _REENTRANT
    #define _REENTRANT
  #endif
#endif

/* Cygwin platform (does not define _WIN32), GNU compiler */
#if defined(__CYGWIN__)
  #undef XSLTPUBFUN
  #undef XSLTPUBVAR
  #undef XSLTCALL
  #if defined(IN_LIBXSLT) && !defined(LIBXSLT_STATIC)
    #define XSLTPUBFUN __declspec(dllexport)
    #define XSLTPUBVAR __declspec(dllexport)
  #else
    #define XSLTPUBFUN
    #if !defined(LIBXSLT_STATIC)
      #define XSLTPUBVAR __declspec(dllimport) extern
    #else
      #define XSLTPUBVAR extern
    #endif
  #endif
  #define XSLTCALL __cdecl
#endif

/* Compatibility */
#if !defined(LIBXSLT_PUBLIC)
#define LIBXSLT_PUBLIC XSLTPUBVAR
#endif

#endif /* __XSLT_EXPORTS_H__ */


