/*
 * Summary: set of routines to process strings
 * Description: type and interfaces needed for the internal string handling
 *              of the library, especially UTF8 processing.
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */

#ifndef __XML_STRING_H__
#define __XML_STRING_H__

#include <stdarg.h>
#include <libxml/xmlversion.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xmlChar:
 *
 * This is a basic byte in an UTF-8 encoded string.
 * It's unsigned allowing to pinpoint case where char * are assigned
 * to xmlChar * (possibly making serialization back impossible).
 */
typedef unsigned char xmlChar;

/**
 * BAD_CAST:
 *
 * Macro to cast a string to an xmlChar * when one know its safe.
 */
#define BAD_CAST (xmlChar *)

/*
 * xmlChar handling
 */
XMLPUBFUN xmlChar * XMLCALL
                xmlStrdup                (const xmlChar *cur);
XMLPUBFUN xmlChar * XMLCALL
                xmlStrndup               (const xmlChar *cur,
                                         int len);
XMLPUBFUN xmlChar * XMLCALL
                xmlCharStrndup           (const char *cur,
                                         int len);
XMLPUBFUN xmlChar * XMLCALL
                xmlCharStrdup            (const char *cur);
XMLPUBFUN xmlChar * XMLCALL
                xmlStrsub                (const xmlChar *str,
                                         int start,
                                         int len);
XMLPUBFUN const xmlChar * XMLCALL
                xmlStrchr                (const xmlChar *str,
                                         xmlChar val);
XMLPUBFUN const xmlChar * XMLCALL
                xmlStrstr                (const xmlChar *str,
                                         const xmlChar *val);
XMLPUBFUN const xmlChar * XMLCALL
                xmlStrcasestr            (const xmlChar *str,
                                         const xmlChar *val);
XMLPUBFUN int XMLCALL
                xmlStrcmp                (const xmlChar *str1,
                                         const xmlChar *str2);
XMLPUBFUN int XMLCALL
                xmlStrncmp               (const xmlChar *str1,
                                         const xmlChar *str2,
                                         int len);
XMLPUBFUN int XMLCALL
                xmlStrcasecmp            (const xmlChar *str1,
                                         const xmlChar *str2);
XMLPUBFUN int XMLCALL
                xmlStrncasecmp           (const xmlChar *str1,
                                         const xmlChar *str2,
                                         int len);
XMLPUBFUN int XMLCALL
                xmlStrEqual              (const xmlChar *str1,
                                         const xmlChar *str2);
XMLPUBFUN int XMLCALL
                xmlStrQEqual             (const xmlChar *pref,
                                         const xmlChar *name,
                                         const xmlChar *str);
XMLPUBFUN int XMLCALL
                xmlStrlen                (const xmlChar *str);
XMLPUBFUN xmlChar * XMLCALL
                xmlStrcat                (xmlChar *cur,
                                         const xmlChar *add);
XMLPUBFUN xmlChar * XMLCALL
                xmlStrncat               (xmlChar *cur,
                                         const xmlChar *add,
                                         int len);
XMLPUBFUN xmlChar * XMLCALL
                xmlStrncatNew            (const xmlChar *str1,
                                         const xmlChar *str2,
                                         int len);
XMLPUBFUN int XMLCALL
                xmlStrPrintf             (xmlChar *buf,
                                         int len,
                                         const char *msg,
                                         ...) LIBXML_ATTR_FORMAT(3,4);
XMLPUBFUN int XMLCALL
                xmlStrVPrintf                (xmlChar *buf,
                                         int len,
                                         const char *msg,
                                         va_list ap) LIBXML_ATTR_FORMAT(3,0);

XMLPUBFUN int XMLCALL
        xmlGetUTF8Char                   (const unsigned char *utf,
                                         int *len);
XMLPUBFUN int XMLCALL
        xmlCheckUTF8                     (const unsigned char *utf);
XMLPUBFUN int XMLCALL
        xmlUTF8Strsize                   (const xmlChar *utf,
                                         int len);
XMLPUBFUN xmlChar * XMLCALL
        xmlUTF8Strndup                   (const xmlChar *utf,
                                         int len);
XMLPUBFUN const xmlChar * XMLCALL
        xmlUTF8Strpos                    (const xmlChar *utf,
                                         int pos);
XMLPUBFUN int XMLCALL
        xmlUTF8Strloc                    (const xmlChar *utf,
                                         const xmlChar *utfchar);
XMLPUBFUN xmlChar * XMLCALL
        xmlUTF8Strsub                    (const xmlChar *utf,
                                         int start,
                                         int len);
XMLPUBFUN int XMLCALL
        xmlUTF8Strlen                    (const xmlChar *utf);
XMLPUBFUN int XMLCALL
        xmlUTF8Size                      (const xmlChar *utf);
XMLPUBFUN int XMLCALL
        xmlUTF8Charcmp                   (const xmlChar *utf1,
                                         const xmlChar *utf2);

#ifdef __cplusplus
}
#endif
#endif /* __XML_STRING_H__ */
