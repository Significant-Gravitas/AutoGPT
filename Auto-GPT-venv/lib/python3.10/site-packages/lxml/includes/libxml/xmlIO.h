/*
 * Summary: interface for the I/O interfaces used by the parser
 * Description: interface for the I/O interfaces used by the parser
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */

#ifndef __XML_IO_H__
#define __XML_IO_H__

#include <stdio.h>
#include <libxml/xmlversion.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Those are the functions and datatypes for the parser input
 * I/O structures.
 */

/**
 * xmlInputMatchCallback:
 * @filename: the filename or URI
 *
 * Callback used in the I/O Input API to detect if the current handler
 * can provide input functionality for this resource.
 *
 * Returns 1 if yes and 0 if another Input module should be used
 */
typedef int (XMLCALL *xmlInputMatchCallback) (char const *filename);
/**
 * xmlInputOpenCallback:
 * @filename: the filename or URI
 *
 * Callback used in the I/O Input API to open the resource
 *
 * Returns an Input context or NULL in case or error
 */
typedef void * (XMLCALL *xmlInputOpenCallback) (char const *filename);
/**
 * xmlInputReadCallback:
 * @context:  an Input context
 * @buffer:  the buffer to store data read
 * @len:  the length of the buffer in bytes
 *
 * Callback used in the I/O Input API to read the resource
 *
 * Returns the number of bytes read or -1 in case of error
 */
typedef int (XMLCALL *xmlInputReadCallback) (void * context, char * buffer, int len);
/**
 * xmlInputCloseCallback:
 * @context:  an Input context
 *
 * Callback used in the I/O Input API to close the resource
 *
 * Returns 0 or -1 in case of error
 */
typedef int (XMLCALL *xmlInputCloseCallback) (void * context);

#ifdef LIBXML_OUTPUT_ENABLED
/*
 * Those are the functions and datatypes for the library output
 * I/O structures.
 */

/**
 * xmlOutputMatchCallback:
 * @filename: the filename or URI
 *
 * Callback used in the I/O Output API to detect if the current handler
 * can provide output functionality for this resource.
 *
 * Returns 1 if yes and 0 if another Output module should be used
 */
typedef int (XMLCALL *xmlOutputMatchCallback) (char const *filename);
/**
 * xmlOutputOpenCallback:
 * @filename: the filename or URI
 *
 * Callback used in the I/O Output API to open the resource
 *
 * Returns an Output context or NULL in case or error
 */
typedef void * (XMLCALL *xmlOutputOpenCallback) (char const *filename);
/**
 * xmlOutputWriteCallback:
 * @context:  an Output context
 * @buffer:  the buffer of data to write
 * @len:  the length of the buffer in bytes
 *
 * Callback used in the I/O Output API to write to the resource
 *
 * Returns the number of bytes written or -1 in case of error
 */
typedef int (XMLCALL *xmlOutputWriteCallback) (void * context, const char * buffer,
                                       int len);
/**
 * xmlOutputCloseCallback:
 * @context:  an Output context
 *
 * Callback used in the I/O Output API to close the resource
 *
 * Returns 0 or -1 in case of error
 */
typedef int (XMLCALL *xmlOutputCloseCallback) (void * context);
#endif /* LIBXML_OUTPUT_ENABLED */

#ifdef __cplusplus
}
#endif

#include <libxml/globals.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/encoding.h>

#ifdef __cplusplus
extern "C" {
#endif
struct _xmlParserInputBuffer {
    void*                  context;
    xmlInputReadCallback   readcallback;
    xmlInputCloseCallback  closecallback;

    xmlCharEncodingHandlerPtr encoder; /* I18N conversions to UTF-8 */

    xmlBufPtr buffer;    /* Local buffer encoded in UTF-8 */
    xmlBufPtr raw;       /* if encoder != NULL buffer for raw input */
    int	compressed;	    /* -1=unknown, 0=not compressed, 1=compressed */
    int error;
    unsigned long rawconsumed;/* amount consumed from raw */
};


#ifdef LIBXML_OUTPUT_ENABLED
struct _xmlOutputBuffer {
    void*                   context;
    xmlOutputWriteCallback  writecallback;
    xmlOutputCloseCallback  closecallback;

    xmlCharEncodingHandlerPtr encoder; /* I18N conversions to UTF-8 */

    xmlBufPtr buffer;    /* Local buffer encoded in UTF-8 or ISOLatin */
    xmlBufPtr conv;      /* if encoder != NULL buffer for output */
    int written;            /* total number of byte written */
    int error;
};
#endif /* LIBXML_OUTPUT_ENABLED */

/*
 * Interfaces for input
 */
XMLPUBFUN void XMLCALL
	xmlCleanupInputCallbacks		(void);

XMLPUBFUN int XMLCALL
	xmlPopInputCallbacks			(void);

XMLPUBFUN void XMLCALL
	xmlRegisterDefaultInputCallbacks	(void);
XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlAllocParserInputBuffer		(xmlCharEncoding enc);

XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlParserInputBufferCreateFilename	(const char *URI,
                                                 xmlCharEncoding enc);
XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlParserInputBufferCreateFile		(FILE *file,
                                                 xmlCharEncoding enc);
XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlParserInputBufferCreateFd		(int fd,
	                                         xmlCharEncoding enc);
XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlParserInputBufferCreateMem		(const char *mem, int size,
	                                         xmlCharEncoding enc);
XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlParserInputBufferCreateStatic	(const char *mem, int size,
	                                         xmlCharEncoding enc);
XMLPUBFUN xmlParserInputBufferPtr XMLCALL
	xmlParserInputBufferCreateIO		(xmlInputReadCallback   ioread,
						 xmlInputCloseCallback  ioclose,
						 void *ioctx,
	                                         xmlCharEncoding enc);
XMLPUBFUN int XMLCALL
	xmlParserInputBufferRead		(xmlParserInputBufferPtr in,
						 int len);
XMLPUBFUN int XMLCALL
	xmlParserInputBufferGrow		(xmlParserInputBufferPtr in,
						 int len);
XMLPUBFUN int XMLCALL
	xmlParserInputBufferPush		(xmlParserInputBufferPtr in,
						 int len,
						 const char *buf);
XMLPUBFUN void XMLCALL
	xmlFreeParserInputBuffer		(xmlParserInputBufferPtr in);
XMLPUBFUN char * XMLCALL
	xmlParserGetDirectory			(const char *filename);

XMLPUBFUN int XMLCALL
	xmlRegisterInputCallbacks		(xmlInputMatchCallback matchFunc,
						 xmlInputOpenCallback openFunc,
						 xmlInputReadCallback readFunc,
						 xmlInputCloseCallback closeFunc);

xmlParserInputBufferPtr
	__xmlParserInputBufferCreateFilename(const char *URI,
						xmlCharEncoding enc);

#ifdef LIBXML_OUTPUT_ENABLED
/*
 * Interfaces for output
 */
XMLPUBFUN void XMLCALL
	xmlCleanupOutputCallbacks		(void);
XMLPUBFUN int XMLCALL
	xmlPopOutputCallbacks			(void);
XMLPUBFUN void XMLCALL
	xmlRegisterDefaultOutputCallbacks(void);
XMLPUBFUN xmlOutputBufferPtr XMLCALL
	xmlAllocOutputBuffer		(xmlCharEncodingHandlerPtr encoder);

XMLPUBFUN xmlOutputBufferPtr XMLCALL
	xmlOutputBufferCreateFilename	(const char *URI,
					 xmlCharEncodingHandlerPtr encoder,
					 int compression);

XMLPUBFUN xmlOutputBufferPtr XMLCALL
	xmlOutputBufferCreateFile	(FILE *file,
					 xmlCharEncodingHandlerPtr encoder);

XMLPUBFUN xmlOutputBufferPtr XMLCALL
	xmlOutputBufferCreateBuffer	(xmlBufferPtr buffer,
					 xmlCharEncodingHandlerPtr encoder);

XMLPUBFUN xmlOutputBufferPtr XMLCALL
	xmlOutputBufferCreateFd		(int fd,
					 xmlCharEncodingHandlerPtr encoder);

XMLPUBFUN xmlOutputBufferPtr XMLCALL
	xmlOutputBufferCreateIO		(xmlOutputWriteCallback   iowrite,
					 xmlOutputCloseCallback  ioclose,
					 void *ioctx,
					 xmlCharEncodingHandlerPtr encoder);

/* Couple of APIs to get the output without digging into the buffers */
XMLPUBFUN const xmlChar * XMLCALL
        xmlOutputBufferGetContent       (xmlOutputBufferPtr out);
XMLPUBFUN size_t XMLCALL
        xmlOutputBufferGetSize          (xmlOutputBufferPtr out);

XMLPUBFUN int XMLCALL
	xmlOutputBufferWrite		(xmlOutputBufferPtr out,
					 int len,
					 const char *buf);
XMLPUBFUN int XMLCALL
	xmlOutputBufferWriteString	(xmlOutputBufferPtr out,
					 const char *str);
XMLPUBFUN int XMLCALL
	xmlOutputBufferWriteEscape	(xmlOutputBufferPtr out,
					 const xmlChar *str,
					 xmlCharEncodingOutputFunc escaping);

XMLPUBFUN int XMLCALL
	xmlOutputBufferFlush		(xmlOutputBufferPtr out);
XMLPUBFUN int XMLCALL
	xmlOutputBufferClose		(xmlOutputBufferPtr out);

XMLPUBFUN int XMLCALL
	xmlRegisterOutputCallbacks	(xmlOutputMatchCallback matchFunc,
					 xmlOutputOpenCallback openFunc,
					 xmlOutputWriteCallback writeFunc,
					 xmlOutputCloseCallback closeFunc);

xmlOutputBufferPtr
	__xmlOutputBufferCreateFilename(const char *URI,
                              xmlCharEncodingHandlerPtr encoder,
                              int compression);

#ifdef LIBXML_HTTP_ENABLED
/*  This function only exists if HTTP support built into the library  */
XMLPUBFUN void XMLCALL
	xmlRegisterHTTPPostCallbacks	(void );
#endif /* LIBXML_HTTP_ENABLED */

#endif /* LIBXML_OUTPUT_ENABLED */

XMLPUBFUN xmlParserInputPtr XMLCALL
	xmlCheckHTTPInput		(xmlParserCtxtPtr ctxt,
					 xmlParserInputPtr ret);

/*
 * A predefined entity loader disabling network accesses
 */
XMLPUBFUN xmlParserInputPtr XMLCALL
	xmlNoNetExternalEntityLoader	(const char *URL,
					 const char *ID,
					 xmlParserCtxtPtr ctxt);

/*
 * xmlNormalizeWindowsPath is obsolete, don't use it.
 * Check xmlCanonicPath in uri.h for a better alternative.
 */
XMLPUBFUN xmlChar * XMLCALL
	xmlNormalizeWindowsPath		(const xmlChar *path);

XMLPUBFUN int XMLCALL
	xmlCheckFilename		(const char *path);
/**
 * Default 'file://' protocol callbacks
 */
XMLPUBFUN int XMLCALL
	xmlFileMatch			(const char *filename);
XMLPUBFUN void * XMLCALL
	xmlFileOpen			(const char *filename);
XMLPUBFUN int XMLCALL
	xmlFileRead			(void * context,
					 char * buffer,
					 int len);
XMLPUBFUN int XMLCALL
	xmlFileClose			(void * context);

/**
 * Default 'http://' protocol callbacks
 */
#ifdef LIBXML_HTTP_ENABLED
XMLPUBFUN int XMLCALL
	xmlIOHTTPMatch			(const char *filename);
XMLPUBFUN void * XMLCALL
	xmlIOHTTPOpen			(const char *filename);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN void * XMLCALL
	xmlIOHTTPOpenW			(const char * post_uri,
					 int   compression );
#endif /* LIBXML_OUTPUT_ENABLED */
XMLPUBFUN int XMLCALL
	xmlIOHTTPRead			(void * context,
					 char * buffer,
					 int len);
XMLPUBFUN int XMLCALL
	xmlIOHTTPClose			(void * context);
#endif /* LIBXML_HTTP_ENABLED */

/**
 * Default 'ftp://' protocol callbacks
 */
#ifdef LIBXML_FTP_ENABLED
XMLPUBFUN int XMLCALL
	xmlIOFTPMatch			(const char *filename);
XMLPUBFUN void * XMLCALL
	xmlIOFTPOpen			(const char *filename);
XMLPUBFUN int XMLCALL
	xmlIOFTPRead			(void * context,
					 char * buffer,
					 int len);
XMLPUBFUN int XMLCALL
	xmlIOFTPClose			(void * context);
#endif /* LIBXML_FTP_ENABLED */

#ifdef __cplusplus
}
#endif

#endif /* __XML_IO_H__ */
