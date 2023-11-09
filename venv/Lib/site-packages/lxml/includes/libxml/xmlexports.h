/*
 * Summary: macros for marking symbols as exportable/importable.
 * Description: macros for marking symbols as exportable/importable.
 *
 * Copy: See Copyright for the status of this software.
 */

#ifndef __XML_EXPORTS_H__
#define __XML_EXPORTS_H__

#if defined(_WIN32) || defined(__CYGWIN__)
/** DOC_DISABLE */

#ifdef LIBXML_STATIC
  #define XMLPUBLIC
#elif defined(IN_LIBXML)
  #define XMLPUBLIC __declspec(dllexport)
#else
  #define XMLPUBLIC __declspec(dllimport)
#endif

#if defined(LIBXML_FASTCALL)
  #define XMLCALL __fastcall
#else
  #define XMLCALL __cdecl
#endif
#define XMLCDECL __cdecl

/** DOC_ENABLE */
#else /* not Windows */

/**
 * XMLPUBLIC:
 *
 * Macro which declares a public symbol
 */
#define XMLPUBLIC

/**
 * XMLCALL:
 *
 * Macro which declares the calling convention for exported functions
 */
#define XMLCALL

/**
 * XMLCDECL:
 *
 * Macro which declares the calling convention for exported functions that
 * use '...'.
 */
#define XMLCDECL

#endif /* platform switch */

/*
 * XMLPUBFUN:
 *
 * Macro which declares an exportable function
 */
#define XMLPUBFUN XMLPUBLIC

/**
 * XMLPUBVAR:
 *
 * Macro which declares an exportable variable
 */
#define XMLPUBVAR XMLPUBLIC extern

/* Compatibility */
#if !defined(LIBXML_DLL_IMPORT)
#define LIBXML_DLL_IMPORT XMLPUBVAR
#endif

#endif /* __XML_EXPORTS_H__ */


