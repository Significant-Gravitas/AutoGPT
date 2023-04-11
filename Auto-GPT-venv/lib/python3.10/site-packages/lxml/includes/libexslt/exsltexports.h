/*
 * Summary: macros for marking symbols as exportable/importable.
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Igor Zlatkovic <igor@zlatkovic.com>
 */

#ifndef __EXSLT_EXPORTS_H__
#define __EXSLT_EXPORTS_H__

/**
 * EXSLTPUBFUN, EXSLTPUBVAR, EXSLTCALL
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
 * EXSLTPUBFUN:
 *
 * Macros which declare an exportable function
 */
#define EXSLTPUBFUN
/**
 * EXSLTPUBVAR:
 *
 * Macros which declare an exportable variable
 */
#define EXSLTPUBVAR extern
/**
 * EXSLTCALL:
 *
 * Macros which declare the called convention for exported functions
 */
#define EXSLTCALL

/** DOC_DISABLE */

/* Windows platform with MS compiler */
#if defined(_WIN32) && defined(_MSC_VER)
  #undef EXSLTPUBFUN
  #undef EXSLTPUBVAR
  #undef EXSLTCALL
  #if defined(IN_LIBEXSLT) && !defined(LIBEXSLT_STATIC)
    #define EXSLTPUBFUN __declspec(dllexport)
    #define EXSLTPUBVAR __declspec(dllexport)
  #else
    #define EXSLTPUBFUN
    #if !defined(LIBEXSLT_STATIC)
      #define EXSLTPUBVAR __declspec(dllimport) extern
    #else
      #define EXSLTPUBVAR extern
    #endif
  #endif
  #define EXSLTCALL __cdecl
  #if !defined _REENTRANT
    #define _REENTRANT
  #endif
#endif

/* Windows platform with Borland compiler */
#if defined(_WIN32) && defined(__BORLANDC__)
  #undef EXSLTPUBFUN
  #undef EXSLTPUBVAR
  #undef EXSLTCALL
  #if defined(IN_LIBEXSLT) && !defined(LIBEXSLT_STATIC)
    #define EXSLTPUBFUN __declspec(dllexport)
    #define EXSLTPUBVAR __declspec(dllexport) extern
  #else
    #define EXSLTPUBFUN
    #if !defined(LIBEXSLT_STATIC)
      #define EXSLTPUBVAR __declspec(dllimport) extern
    #else
      #define EXSLTPUBVAR extern
    #endif
  #endif
  #define EXSLTCALL __cdecl
  #if !defined _REENTRANT
    #define _REENTRANT
  #endif
#endif

/* Windows platform with GNU compiler (Mingw) */
#if defined(_WIN32) && defined(__MINGW32__)
  #undef EXSLTPUBFUN
  #undef EXSLTPUBVAR
  #undef EXSLTCALL
/*
  #if defined(IN_LIBEXSLT) && !defined(LIBEXSLT_STATIC)
*/
  #if !defined(LIBEXSLT_STATIC)
    #define EXSLTPUBFUN __declspec(dllexport)
    #define EXSLTPUBVAR __declspec(dllexport) extern
  #else
    #define EXSLTPUBFUN
    #if !defined(LIBEXSLT_STATIC)
      #define EXSLTPUBVAR __declspec(dllimport) extern
    #else
      #define EXSLTPUBVAR extern
    #endif
  #endif
  #define EXSLTCALL __cdecl
  #if !defined _REENTRANT
    #define _REENTRANT
  #endif
#endif

/* Cygwin platform (does not define _WIN32), GNU compiler */
#if defined(__CYGWIN__)
  #undef EXSLTPUBFUN
  #undef EXSLTPUBVAR
  #undef EXSLTCALL
  #if defined(IN_LIBEXSLT) && !defined(LIBEXSLT_STATIC)
    #define EXSLTPUBFUN __declspec(dllexport)
    #define EXSLTPUBVAR __declspec(dllexport)
  #else
    #define EXSLTPUBFUN
    #if !defined(LIBEXSLT_STATIC)
      #define EXSLTPUBVAR __declspec(dllimport) extern
    #else
      #define EXSLTPUBVAR extern
    #endif
  #endif
  #define EXSLTCALL __cdecl
#endif

/* Compatibility */
#if !defined(LIBEXSLT_PUBLIC)
#define LIBEXSLT_PUBLIC EXSLTPUBVAR
#endif

#endif /* __EXSLT_EXPORTS_H__ */


