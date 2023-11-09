/*
 * Summary: Windows configuration header
 * Description: Windows configuration header
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Igor Zlatkovic
 */
#ifndef __LIBXSLT_WIN32_CONFIG__
#define __LIBXSLT_WIN32_CONFIG__

#define HAVE_CTYPE_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STDARG_H 1
#define HAVE_MALLOC_H 1
#define HAVE_TIME_H 1
#define HAVE_LOCALTIME 1
#define HAVE_GMTIME 1
#define HAVE_TIME 1
#define HAVE_MATH_H 1
#define HAVE_FCNTL_H 1

#include <io.h>

#define HAVE_ISINF
#define HAVE_ISNAN

#include <math.h>
#if defined _MSC_VER || defined __MINGW32__
/* MS C-runtime has functions which can be used in order to determine if
   a given floating-point variable contains NaN, (+-)INF. These are
   preferred, because floating-point technology is considered propriatary
   by MS and we can assume that their functions know more about their
   oddities than we do. */
#include <float.h>
/* Bjorn Reese figured a quite nice construct for isinf() using the
   _fpclass() function. */
#ifndef isinf
#define isinf(d) ((_fpclass(d) == _FPCLASS_PINF) ? 1 \
	: ((_fpclass(d) == _FPCLASS_NINF) ? -1 : 0))
#endif
/* _isnan(x) returns nonzero if (x == NaN) and zero otherwise. */
#ifndef isnan
#define isnan(d) (_isnan(d))
#endif
#else /* _MSC_VER */
static int isinf (double d) {
    int expon = 0;
    double val = frexp (d, &expon);
    if (expon == 1025) {
        if (val == 0.5) {
            return 1;
        } else if (val == -0.5) {
            return -1;
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}
static int isnan (double d) {
    int expon = 0;
    double val = frexp (d, &expon);
    if (expon == 1025) {
        if (val == 0.5) {
            return 0;
        } else if (val == -0.5) {
            return 0;
        } else {
            return 1;
        }
    } else {
        return 0;
    }
}
#endif /* _MSC_VER */

#include <direct.h>

/* snprintf emulation taken from http://stackoverflow.com/a/8712996/1956010 */
#if defined(_MSC_VER) && _MSC_VER < 1900

#include <stdarg.h>
#include <stdio.h>

#define snprintf c99_snprintf
#define vsnprintf c99_vsnprintf

__inline int c99_vsnprintf(char *outBuf, size_t size, const char *format, va_list ap)
{
    int count = -1;

    if (size != 0)
        count = _vsnprintf_s(outBuf, size, _TRUNCATE, format, ap);
    if (count == -1)
        count = _vscprintf(format, ap);

    return count;
}

__inline int c99_snprintf(char *outBuf, size_t size, const char *format, ...)
{
    int count;
    va_list ap;

    va_start(ap, format);
    count = c99_vsnprintf(outBuf, size, format, ap);
    va_end(ap);

    return count;
}

#endif /* defined(_MSC_VER) && _MSC_VER < 1900 */

#define HAVE_SYS_STAT_H
#define HAVE__STAT
#define HAVE_STRING_H

#include <libxml/xmlversion.h>

#ifndef ATTRIBUTE_UNUSED
#define ATTRIBUTE_UNUSED
#endif

#define _WINSOCKAPI_

#endif /* __LIBXSLT_WIN32_CONFIG__ */

