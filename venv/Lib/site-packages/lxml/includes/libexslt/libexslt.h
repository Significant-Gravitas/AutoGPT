/*
 * libexslt.h: internal header only used during the compilation of libexslt
 *
 * See COPYRIGHT for the status of this software
 *
 * Author: daniel@veillard.com
 */

#ifndef __XSLT_LIBEXSLT_H__
#define __XSLT_LIBEXSLT_H__

#if defined(_WIN32) && !defined (__CYGWIN__) && !defined (__MINGW32__)
#include <win32config.h>
#else
#include "config.h"
#endif

#include <libxslt/xsltconfig.h>
#include <libxml/xmlversion.h>

#if !defined LIBEXSLT_PUBLIC
#if (defined (__CYGWIN__) || defined _MSC_VER) && !defined IN_LIBEXSLT && !defined LIBEXSLT_STATIC
#define LIBEXSLT_PUBLIC __declspec(dllimport)
#else
#define LIBEXSLT_PUBLIC
#endif
#endif

#endif /* ! __XSLT_LIBEXSLT_H__ */
