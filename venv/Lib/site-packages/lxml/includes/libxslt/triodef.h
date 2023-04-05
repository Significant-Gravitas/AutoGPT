/*************************************************************************
 *
 * $Id$
 *
 * Copyright (C) 2001 Bjorn Reese <breese@users.sourceforge.net>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF
 * MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS AND
 * CONTRIBUTORS ACCEPT NO RESPONSIBILITY IN ANY CONCEIVABLE MANNER.
 *
 ************************************************************************/

#ifndef TRIO_TRIODEF_H
#define TRIO_TRIODEF_H

/*************************************************************************
 * Platform and compiler support detection
 */
#if defined(__GNUC__)
# define TRIO_COMPILER_GCC
#elif defined(__SUNPRO_C)
# define TRIO_COMPILER_SUNPRO
#elif defined(__SUNPRO_CC)
# define TRIO_COMPILER_SUNPRO
# define __SUNPRO_C __SUNPRO_CC
#elif defined(__xlC__) || defined(__IBMC__) || defined(__IBMCPP__)
# define TRIO_COMPILER_XLC
#elif defined(_AIX) && !defined(__GNUC__)
# define TRIO_COMPILER_XLC /* Workaround for old xlc */
#elif defined(__DECC) || defined(__DECCXX)
# define TRIO_COMPILER_DECC
#elif defined(__osf__) && defined(__LANGUAGE_C__)
# define TRIO_COMPILER_DECC /* Workaround for old DEC C compilers */
#elif defined(_MSC_VER)
# define TRIO_COMPILER_MSVC
#elif defined(__BORLANDC__)
# define TRIO_COMPILER_BCB
#endif

#if defined(VMS) || defined(__VMS)
/*
 * VMS is placed first to avoid identifying the platform as Unix
 * based on the DECC compiler later on.
 */
# define TRIO_PLATFORM_VMS
#elif defined(unix) || defined(__unix) || defined(__unix__)
# define TRIO_PLATFORM_UNIX
#elif defined(TRIO_COMPILER_XLC) || defined(_AIX)
# define TRIO_PLATFORM_UNIX
#elif defined(TRIO_COMPILER_DECC) || defined(__osf___)
# define TRIO_PLATFORM_UNIX
#elif defined(__NetBSD__)
# define TRIO_PLATFORM_UNIX
#elif defined(__QNX__)
# define TRIO_PLATFORM_UNIX
# define TRIO_PLATFORM_QNX
#elif defined(__CYGWIN__)
# define TRIO_PLATFORM_UNIX
#elif defined(AMIGA) && defined(TRIO_COMPILER_GCC)
# define TRIO_PLATFORM_UNIX
#elif defined(TRIO_COMPILER_MSVC) || defined(WIN32) || defined(_WIN32)
# define TRIO_PLATFORM_WIN32
#elif defined(mpeix) || defined(__mpexl)
# define TRIO_PLATFORM_MPEIX
#endif

#if defined(_AIX)
# define TRIO_PLATFORM_AIX
#elif defined(__hpux)
# define TRIO_PLATFORM_HPUX
#elif defined(sun) || defined(__sun__)
# if defined(__SVR4) || defined(__svr4__)
#  define TRIO_PLATFORM_SOLARIS
# else
#  define TRIO_PLATFORM_SUNOS
# endif
#endif

#if defined(__STDC__) || defined(TRIO_COMPILER_MSVC) || defined(TRIO_COMPILER_BCB)
# define TRIO_COMPILER_SUPPORTS_C89
# if defined(__STDC_VERSION__)
#  define TRIO_COMPILER_SUPPORTS_C90
#  if (__STDC_VERSION__ >= 199409L)
#   define TRIO_COMPILER_SUPPORTS_C94
#  endif
#  if (__STDC_VERSION__ >= 199901L)
#   define TRIO_COMPILER_SUPPORTS_C99
#  endif
# elif defined(TRIO_COMPILER_SUNPRO)
#  if (__SUNPRO_C >= 0x420)
#   define TRIO_COMPILER_SUPPORTS_C94
#  endif
# endif
#endif

#if defined(_XOPEN_SOURCE)
# if defined(_XOPEN_SOURCE_EXTENDED)
#  define TRIO_COMPILER_SUPPORTS_UNIX95
# endif
# if (_XOPEN_VERSION >= 500)
#  define TRIO_COMPILER_SUPPORTS_UNIX98
# endif
# if (_XOPEN_VERSION >= 600)
#  define TRIO_COMPILER_SUPPORTS_UNIX01
# endif
#endif

/*************************************************************************
 * Generic defines
 */

#if !defined(TRIO_PUBLIC)
# define TRIO_PUBLIC
#endif
#if !defined(TRIO_PRIVATE)
# define TRIO_PRIVATE static
#endif

#if !(defined(TRIO_COMPILER_SUPPORTS_C89) || defined(__cplusplus))
# define TRIO_COMPILER_ANCIENT
#endif

#if defined(TRIO_COMPILER_ANCIENT)
# define TRIO_CONST
# define TRIO_VOLATILE
# define TRIO_SIGNED
typedef double trio_long_double_t;
typedef char * trio_pointer_t;
# define TRIO_SUFFIX_LONG(x) x
# define TRIO_PROTO(x) ()
# define TRIO_NOARGS
# define TRIO_ARGS1(list,a1) list a1;
# define TRIO_ARGS2(list,a1,a2) list a1; a2;
# define TRIO_ARGS3(list,a1,a2,a3) list a1; a2; a3;
# define TRIO_ARGS4(list,a1,a2,a3,a4) list a1; a2; a3; a4;
# define TRIO_ARGS5(list,a1,a2,a3,a4,a5) list a1; a2; a3; a4; a5;
# define TRIO_ARGS6(list,a1,a2,a3,a4,a5,a6) list a1; a2; a3; a4; a5; a6;
# define TRIO_VARGS2(list,a1,a2) list a1; a2
# define TRIO_VARGS3(list,a1,a2,a3) list a1; a2; a3
# define TRIO_VARGS4(list,a1,a2,a3,a4) list a1; a2; a3; a4
# define TRIO_VARGS5(list,a1,a2,a3,a4,a5) list a1; a2; a3; a4; a5
# define TRIO_VA_DECL va_dcl
# define TRIO_VA_START(x,y) va_start(x)
# define TRIO_VA_END(x) va_end(x)
#else /* ANSI C */
# define TRIO_CONST const
# define TRIO_VOLATILE volatile
# define TRIO_SIGNED signed
typedef long double trio_long_double_t;
typedef void * trio_pointer_t;
# define TRIO_SUFFIX_LONG(x) x ## L
# define TRIO_PROTO(x) x
# define TRIO_NOARGS void
# define TRIO_ARGS1(list,a1) (a1)
# define TRIO_ARGS2(list,a1,a2) (a1,a2)
# define TRIO_ARGS3(list,a1,a2,a3) (a1,a2,a3)
# define TRIO_ARGS4(list,a1,a2,a3,a4) (a1,a2,a3,a4)
# define TRIO_ARGS5(list,a1,a2,a3,a4,a5) (a1,a2,a3,a4,a5)
# define TRIO_ARGS6(list,a1,a2,a3,a4,a5,a6) (a1,a2,a3,a4,a5,a6)
# define TRIO_VARGS2 TRIO_ARGS2
# define TRIO_VARGS3 TRIO_ARGS3
# define TRIO_VARGS4 TRIO_ARGS4
# define TRIO_VARGS5 TRIO_ARGS5
# define TRIO_VA_DECL ...
# define TRIO_VA_START(x,y) va_start(x,y)
# define TRIO_VA_END(x) va_end(x)
#endif

#if defined(TRIO_COMPILER_SUPPORTS_C99) || defined(__cplusplus)
# define TRIO_INLINE inline
#elif defined(TRIO_COMPILER_GCC)
# define TRIO_INLINE __inline__
#elif defined(TRIO_COMPILER_MSVC)
# define TRIO_INLINE _inline
#elif defined(TRIO_COMPILER_BCB)
# define TRIO_INLINE __inline
#else
# define TRIO_INLINE
#endif

/*************************************************************************
 * Workarounds
 */

#if defined(TRIO_PLATFORM_VMS)
/*
 * Computations done with constants at compile time can trigger these
 * even when compiling with IEEE enabled.
 */
# pragma message disable (UNDERFLOW, FLOATOVERFL)

# if (__CRTL_VER < 80000000)
/*
 * Although the compiler supports C99 language constructs, the C
 * run-time library does not contain all C99 functions.
 *
 * This was the case for 70300022. Update the 80000000 value when
 * it has been accurately determined what version of the library
 * supports C99.
 */
#  if defined(TRIO_COMPILER_SUPPORTS_C99)
#   undef TRIO_COMPILER_SUPPORTS_C99
#  endif
# endif
#endif

/*
 * Not all preprocessors supports the LL token.
 */
#if defined(TRIO_COMPILER_BCB)
#else
# define TRIO_COMPILER_SUPPORTS_LL
#endif

#endif /* TRIO_TRIODEF_H */
