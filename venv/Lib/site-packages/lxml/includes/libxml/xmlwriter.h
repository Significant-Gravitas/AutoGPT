/*
 * Summary: text writing API for XML
 * Description: text writing API for XML
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Alfred Mickautsch <alfred@mickautsch.de>
 */

#ifndef __XML_XMLWRITER_H__
#define __XML_XMLWRITER_H__

#include <libxml/xmlversion.h>

#ifdef LIBXML_WRITER_ENABLED

#include <stdarg.h>
#include <libxml/xmlIO.h>
#include <libxml/list.h>
#include <libxml/xmlstring.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct _xmlTextWriter xmlTextWriter;
    typedef xmlTextWriter *xmlTextWriterPtr;

/*
 * Constructors & Destructor
 */
    XMLPUBFUN xmlTextWriterPtr XMLCALL
        xmlNewTextWriter(xmlOutputBufferPtr out);
    XMLPUBFUN xmlTextWriterPtr XMLCALL
        xmlNewTextWriterFilename(const char *uri, int compression);
    XMLPUBFUN xmlTextWriterPtr XMLCALL
        xmlNewTextWriterMemory(xmlBufferPtr buf, int compression);
    XMLPUBFUN xmlTextWriterPtr XMLCALL
        xmlNewTextWriterPushParser(xmlParserCtxtPtr ctxt, int compression);
    XMLPUBFUN xmlTextWriterPtr XMLCALL
        xmlNewTextWriterDoc(xmlDocPtr * doc, int compression);
    XMLPUBFUN xmlTextWriterPtr XMLCALL
        xmlNewTextWriterTree(xmlDocPtr doc, xmlNodePtr node,
                             int compression);
    XMLPUBFUN void XMLCALL xmlFreeTextWriter(xmlTextWriterPtr writer);

/*
 * Functions
 */


/*
 * Document
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartDocument(xmlTextWriterPtr writer,
                                   const char *version,
                                   const char *encoding,
                                   const char *standalone);
    XMLPUBFUN int XMLCALL xmlTextWriterEndDocument(xmlTextWriterPtr
                                                   writer);

/*
 * Comments
 */
    XMLPUBFUN int XMLCALL xmlTextWriterStartComment(xmlTextWriterPtr
                                                    writer);
    XMLPUBFUN int XMLCALL xmlTextWriterEndComment(xmlTextWriterPtr writer);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatComment(xmlTextWriterPtr writer,
                                        const char *format, ...)
					LIBXML_ATTR_FORMAT(2,3);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatComment(xmlTextWriterPtr writer,
                                         const char *format,
                                         va_list argptr)
					 LIBXML_ATTR_FORMAT(2,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteComment(xmlTextWriterPtr
                                                    writer,
                                                    const xmlChar *
                                                    content);

/*
 * Elements
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartElement(xmlTextWriterPtr writer,
                                  const xmlChar * name);
    XMLPUBFUN int XMLCALL xmlTextWriterStartElementNS(xmlTextWriterPtr
                                                      writer,
                                                      const xmlChar *
                                                      prefix,
                                                      const xmlChar * name,
                                                      const xmlChar *
                                                      namespaceURI);
    XMLPUBFUN int XMLCALL xmlTextWriterEndElement(xmlTextWriterPtr writer);
    XMLPUBFUN int XMLCALL xmlTextWriterFullEndElement(xmlTextWriterPtr
                                                      writer);

/*
 * Elements conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatElement(xmlTextWriterPtr writer,
                                        const xmlChar * name,
                                        const char *format, ...)
					LIBXML_ATTR_FORMAT(3,4);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatElement(xmlTextWriterPtr writer,
                                         const xmlChar * name,
                                         const char *format,
                                         va_list argptr)
					 LIBXML_ATTR_FORMAT(3,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteElement(xmlTextWriterPtr
                                                    writer,
                                                    const xmlChar * name,
                                                    const xmlChar *
                                                    content);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatElementNS(xmlTextWriterPtr writer,
                                          const xmlChar * prefix,
                                          const xmlChar * name,
                                          const xmlChar * namespaceURI,
                                          const char *format, ...)
					  LIBXML_ATTR_FORMAT(5,6);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatElementNS(xmlTextWriterPtr writer,
                                           const xmlChar * prefix,
                                           const xmlChar * name,
                                           const xmlChar * namespaceURI,
                                           const char *format,
                                           va_list argptr)
					   LIBXML_ATTR_FORMAT(5,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteElementNS(xmlTextWriterPtr
                                                      writer,
                                                      const xmlChar *
                                                      prefix,
                                                      const xmlChar * name,
                                                      const xmlChar *
                                                      namespaceURI,
                                                      const xmlChar *
                                                      content);

/*
 * Text
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatRaw(xmlTextWriterPtr writer,
                                    const char *format, ...)
				    LIBXML_ATTR_FORMAT(2,3);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatRaw(xmlTextWriterPtr writer,
                                     const char *format, va_list argptr)
				     LIBXML_ATTR_FORMAT(2,0);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteRawLen(xmlTextWriterPtr writer,
                                 const xmlChar * content, int len);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteRaw(xmlTextWriterPtr writer,
                              const xmlChar * content);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteFormatString(xmlTextWriterPtr
                                                         writer,
                                                         const char
                                                         *format, ...)
							 LIBXML_ATTR_FORMAT(2,3);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteVFormatString(xmlTextWriterPtr
                                                          writer,
                                                          const char
                                                          *format,
                                                          va_list argptr)
							  LIBXML_ATTR_FORMAT(2,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteString(xmlTextWriterPtr writer,
                                                   const xmlChar *
                                                   content);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteBase64(xmlTextWriterPtr writer,
                                                   const char *data,
                                                   int start, int len);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteBinHex(xmlTextWriterPtr writer,
                                                   const char *data,
                                                   int start, int len);

/*
 * Attributes
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartAttribute(xmlTextWriterPtr writer,
                                    const xmlChar * name);
    XMLPUBFUN int XMLCALL xmlTextWriterStartAttributeNS(xmlTextWriterPtr
                                                        writer,
                                                        const xmlChar *
                                                        prefix,
                                                        const xmlChar *
                                                        name,
                                                        const xmlChar *
                                                        namespaceURI);
    XMLPUBFUN int XMLCALL xmlTextWriterEndAttribute(xmlTextWriterPtr
                                                    writer);

/*
 * Attributes conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatAttribute(xmlTextWriterPtr writer,
                                          const xmlChar * name,
                                          const char *format, ...)
					  LIBXML_ATTR_FORMAT(3,4);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatAttribute(xmlTextWriterPtr writer,
                                           const xmlChar * name,
                                           const char *format,
                                           va_list argptr)
					   LIBXML_ATTR_FORMAT(3,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteAttribute(xmlTextWriterPtr
                                                      writer,
                                                      const xmlChar * name,
                                                      const xmlChar *
                                                      content);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatAttributeNS(xmlTextWriterPtr writer,
                                            const xmlChar * prefix,
                                            const xmlChar * name,
                                            const xmlChar * namespaceURI,
                                            const char *format, ...)
					    LIBXML_ATTR_FORMAT(5,6);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatAttributeNS(xmlTextWriterPtr writer,
                                             const xmlChar * prefix,
                                             const xmlChar * name,
                                             const xmlChar * namespaceURI,
                                             const char *format,
                                             va_list argptr)
					     LIBXML_ATTR_FORMAT(5,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteAttributeNS(xmlTextWriterPtr
                                                        writer,
                                                        const xmlChar *
                                                        prefix,
                                                        const xmlChar *
                                                        name,
                                                        const xmlChar *
                                                        namespaceURI,
                                                        const xmlChar *
                                                        content);

/*
 * PI's
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartPI(xmlTextWriterPtr writer,
                             const xmlChar * target);
    XMLPUBFUN int XMLCALL xmlTextWriterEndPI(xmlTextWriterPtr writer);

/*
 * PI conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatPI(xmlTextWriterPtr writer,
                                   const xmlChar * target,
                                   const char *format, ...)
				   LIBXML_ATTR_FORMAT(3,4);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatPI(xmlTextWriterPtr writer,
                                    const xmlChar * target,
                                    const char *format, va_list argptr)
				    LIBXML_ATTR_FORMAT(3,0);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWritePI(xmlTextWriterPtr writer,
                             const xmlChar * target,
                             const xmlChar * content);

/**
 * xmlTextWriterWriteProcessingInstruction:
 *
 * This macro maps to xmlTextWriterWritePI
 */
#define xmlTextWriterWriteProcessingInstruction xmlTextWriterWritePI

/*
 * CDATA
 */
    XMLPUBFUN int XMLCALL xmlTextWriterStartCDATA(xmlTextWriterPtr writer);
    XMLPUBFUN int XMLCALL xmlTextWriterEndCDATA(xmlTextWriterPtr writer);

/*
 * CDATA conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatCDATA(xmlTextWriterPtr writer,
                                      const char *format, ...)
				      LIBXML_ATTR_FORMAT(2,3);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatCDATA(xmlTextWriterPtr writer,
                                       const char *format, va_list argptr)
				       LIBXML_ATTR_FORMAT(2,0);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteCDATA(xmlTextWriterPtr writer,
                                const xmlChar * content);

/*
 * DTD
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartDTD(xmlTextWriterPtr writer,
                              const xmlChar * name,
                              const xmlChar * pubid,
                              const xmlChar * sysid);
    XMLPUBFUN int XMLCALL xmlTextWriterEndDTD(xmlTextWriterPtr writer);

/*
 * DTD conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatDTD(xmlTextWriterPtr writer,
                                    const xmlChar * name,
                                    const xmlChar * pubid,
                                    const xmlChar * sysid,
                                    const char *format, ...)
				    LIBXML_ATTR_FORMAT(5,6);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatDTD(xmlTextWriterPtr writer,
                                     const xmlChar * name,
                                     const xmlChar * pubid,
                                     const xmlChar * sysid,
                                     const char *format, va_list argptr)
				     LIBXML_ATTR_FORMAT(5,0);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteDTD(xmlTextWriterPtr writer,
                              const xmlChar * name,
                              const xmlChar * pubid,
                              const xmlChar * sysid,
                              const xmlChar * subset);

/**
 * xmlTextWriterWriteDocType:
 *
 * this macro maps to xmlTextWriterWriteDTD
 */
#define xmlTextWriterWriteDocType xmlTextWriterWriteDTD

/*
 * DTD element definition
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartDTDElement(xmlTextWriterPtr writer,
                                     const xmlChar * name);
    XMLPUBFUN int XMLCALL xmlTextWriterEndDTDElement(xmlTextWriterPtr
                                                     writer);

/*
 * DTD element definition conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatDTDElement(xmlTextWriterPtr writer,
                                           const xmlChar * name,
                                           const char *format, ...)
					   LIBXML_ATTR_FORMAT(3,4);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatDTDElement(xmlTextWriterPtr writer,
                                            const xmlChar * name,
                                            const char *format,
                                            va_list argptr)
					    LIBXML_ATTR_FORMAT(3,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteDTDElement(xmlTextWriterPtr
                                                       writer,
                                                       const xmlChar *
                                                       name,
                                                       const xmlChar *
                                                       content);

/*
 * DTD attribute list definition
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartDTDAttlist(xmlTextWriterPtr writer,
                                     const xmlChar * name);
    XMLPUBFUN int XMLCALL xmlTextWriterEndDTDAttlist(xmlTextWriterPtr
                                                     writer);

/*
 * DTD attribute list definition conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatDTDAttlist(xmlTextWriterPtr writer,
                                           const xmlChar * name,
                                           const char *format, ...)
					   LIBXML_ATTR_FORMAT(3,4);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatDTDAttlist(xmlTextWriterPtr writer,
                                            const xmlChar * name,
                                            const char *format,
                                            va_list argptr)
					    LIBXML_ATTR_FORMAT(3,0);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteDTDAttlist(xmlTextWriterPtr
                                                       writer,
                                                       const xmlChar *
                                                       name,
                                                       const xmlChar *
                                                       content);

/*
 * DTD entity definition
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterStartDTDEntity(xmlTextWriterPtr writer,
                                    int pe, const xmlChar * name);
    XMLPUBFUN int XMLCALL xmlTextWriterEndDTDEntity(xmlTextWriterPtr
                                                    writer);

/*
 * DTD entity definition conveniency functions
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteFormatDTDInternalEntity(xmlTextWriterPtr writer,
                                                  int pe,
                                                  const xmlChar * name,
                                                  const char *format, ...)
						  LIBXML_ATTR_FORMAT(4,5);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteVFormatDTDInternalEntity(xmlTextWriterPtr writer,
                                                   int pe,
                                                   const xmlChar * name,
                                                   const char *format,
                                                   va_list argptr)
						   LIBXML_ATTR_FORMAT(4,0);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteDTDInternalEntity(xmlTextWriterPtr writer,
                                            int pe,
                                            const xmlChar * name,
                                            const xmlChar * content);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteDTDExternalEntity(xmlTextWriterPtr writer,
                                            int pe,
                                            const xmlChar * name,
                                            const xmlChar * pubid,
                                            const xmlChar * sysid,
                                            const xmlChar * ndataid);
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteDTDExternalEntityContents(xmlTextWriterPtr
                                                    writer,
                                                    const xmlChar * pubid,
                                                    const xmlChar * sysid,
                                                    const xmlChar *
                                                    ndataid);
    XMLPUBFUN int XMLCALL xmlTextWriterWriteDTDEntity(xmlTextWriterPtr
                                                      writer, int pe,
                                                      const xmlChar * name,
                                                      const xmlChar *
                                                      pubid,
                                                      const xmlChar *
                                                      sysid,
                                                      const xmlChar *
                                                      ndataid,
                                                      const xmlChar *
                                                      content);

/*
 * DTD notation definition
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterWriteDTDNotation(xmlTextWriterPtr writer,
                                      const xmlChar * name,
                                      const xmlChar * pubid,
                                      const xmlChar * sysid);

/*
 * Indentation
 */
    XMLPUBFUN int XMLCALL
        xmlTextWriterSetIndent(xmlTextWriterPtr writer, int indent);
    XMLPUBFUN int XMLCALL
        xmlTextWriterSetIndentString(xmlTextWriterPtr writer,
                                     const xmlChar * str);

    XMLPUBFUN int XMLCALL
        xmlTextWriterSetQuoteChar(xmlTextWriterPtr writer, xmlChar quotechar);


/*
 * misc
 */
    XMLPUBFUN int XMLCALL xmlTextWriterFlush(xmlTextWriterPtr writer);

#ifdef __cplusplus
}
#endif

#endif /* LIBXML_WRITER_ENABLED */

#endif                          /* __XML_XMLWRITER_H__ */
