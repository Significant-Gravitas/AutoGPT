/*
 * Summary: implementation of XML Schema Datatypes
 * Description: module providing the XML Schema Datatypes implementation
 *              both definition and validity checking
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */


#ifndef __XML_SCHEMA_TYPES_H__
#define __XML_SCHEMA_TYPES_H__

#include <libxml/xmlversion.h>

#ifdef LIBXML_SCHEMAS_ENABLED

#include <libxml/schemasInternals.h>
#include <libxml/xmlschemas.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    XML_SCHEMA_WHITESPACE_UNKNOWN = 0,
    XML_SCHEMA_WHITESPACE_PRESERVE = 1,
    XML_SCHEMA_WHITESPACE_REPLACE = 2,
    XML_SCHEMA_WHITESPACE_COLLAPSE = 3
} xmlSchemaWhitespaceValueType;

XMLPUBFUN void XMLCALL
		xmlSchemaInitTypes		(void);
XMLPUBFUN void XMLCALL
		xmlSchemaCleanupTypes		(void);
XMLPUBFUN xmlSchemaTypePtr XMLCALL
		xmlSchemaGetPredefinedType	(const xmlChar *name,
						 const xmlChar *ns);
XMLPUBFUN int XMLCALL
		xmlSchemaValidatePredefinedType	(xmlSchemaTypePtr type,
						 const xmlChar *value,
						 xmlSchemaValPtr *val);
XMLPUBFUN int XMLCALL
		xmlSchemaValPredefTypeNode	(xmlSchemaTypePtr type,
						 const xmlChar *value,
						 xmlSchemaValPtr *val,
						 xmlNodePtr node);
XMLPUBFUN int XMLCALL
		xmlSchemaValidateFacet		(xmlSchemaTypePtr base,
						 xmlSchemaFacetPtr facet,
						 const xmlChar *value,
						 xmlSchemaValPtr val);
XMLPUBFUN int XMLCALL
		xmlSchemaValidateFacetWhtsp	(xmlSchemaFacetPtr facet,
						 xmlSchemaWhitespaceValueType fws,
						 xmlSchemaValType valType,
						 const xmlChar *value,
						 xmlSchemaValPtr val,
						 xmlSchemaWhitespaceValueType ws);
XMLPUBFUN void XMLCALL
		xmlSchemaFreeValue		(xmlSchemaValPtr val);
XMLPUBFUN xmlSchemaFacetPtr XMLCALL
		xmlSchemaNewFacet		(void);
XMLPUBFUN int XMLCALL
		xmlSchemaCheckFacet		(xmlSchemaFacetPtr facet,
						 xmlSchemaTypePtr typeDecl,
						 xmlSchemaParserCtxtPtr ctxt,
						 const xmlChar *name);
XMLPUBFUN void XMLCALL
		xmlSchemaFreeFacet		(xmlSchemaFacetPtr facet);
XMLPUBFUN int XMLCALL
		xmlSchemaCompareValues		(xmlSchemaValPtr x,
						 xmlSchemaValPtr y);
XMLPUBFUN xmlSchemaTypePtr XMLCALL
    xmlSchemaGetBuiltInListSimpleTypeItemType	(xmlSchemaTypePtr type);
XMLPUBFUN int XMLCALL
    xmlSchemaValidateListSimpleTypeFacet	(xmlSchemaFacetPtr facet,
						 const xmlChar *value,
						 unsigned long actualLen,
						 unsigned long *expectedLen);
XMLPUBFUN xmlSchemaTypePtr XMLCALL
		xmlSchemaGetBuiltInType		(xmlSchemaValType type);
XMLPUBFUN int XMLCALL
		xmlSchemaIsBuiltInTypeFacet	(xmlSchemaTypePtr type,
						 int facetType);
XMLPUBFUN xmlChar * XMLCALL
		xmlSchemaCollapseString		(const xmlChar *value);
XMLPUBFUN xmlChar * XMLCALL
		xmlSchemaWhiteSpaceReplace	(const xmlChar *value);
XMLPUBFUN unsigned long  XMLCALL
		xmlSchemaGetFacetValueAsULong	(xmlSchemaFacetPtr facet);
XMLPUBFUN int XMLCALL
		xmlSchemaValidateLengthFacet	(xmlSchemaTypePtr type,
						 xmlSchemaFacetPtr facet,
						 const xmlChar *value,
						 xmlSchemaValPtr val,
						 unsigned long *length);
XMLPUBFUN int XMLCALL
		xmlSchemaValidateLengthFacetWhtsp(xmlSchemaFacetPtr facet,
						  xmlSchemaValType valType,
						  const xmlChar *value,
						  xmlSchemaValPtr val,
						  unsigned long *length,
						  xmlSchemaWhitespaceValueType ws);
XMLPUBFUN int XMLCALL
		xmlSchemaValPredefTypeNodeNoNorm(xmlSchemaTypePtr type,
						 const xmlChar *value,
						 xmlSchemaValPtr *val,
						 xmlNodePtr node);
XMLPUBFUN int XMLCALL
		xmlSchemaGetCanonValue		(xmlSchemaValPtr val,
						 const xmlChar **retValue);
XMLPUBFUN int XMLCALL
		xmlSchemaGetCanonValueWhtsp	(xmlSchemaValPtr val,
						 const xmlChar **retValue,
						 xmlSchemaWhitespaceValueType ws);
XMLPUBFUN int XMLCALL
		xmlSchemaValueAppend		(xmlSchemaValPtr prev,
						 xmlSchemaValPtr cur);
XMLPUBFUN xmlSchemaValPtr XMLCALL
		xmlSchemaValueGetNext		(xmlSchemaValPtr cur);
XMLPUBFUN const xmlChar * XMLCALL
		xmlSchemaValueGetAsString	(xmlSchemaValPtr val);
XMLPUBFUN int XMLCALL
		xmlSchemaValueGetAsBoolean	(xmlSchemaValPtr val);
XMLPUBFUN xmlSchemaValPtr XMLCALL
		xmlSchemaNewStringValue		(xmlSchemaValType type,
						 const xmlChar *value);
XMLPUBFUN xmlSchemaValPtr XMLCALL
		xmlSchemaNewNOTATIONValue	(const xmlChar *name,
						 const xmlChar *ns);
XMLPUBFUN xmlSchemaValPtr XMLCALL
		xmlSchemaNewQNameValue		(const xmlChar *namespaceName,
						 const xmlChar *localName);
XMLPUBFUN int XMLCALL
		xmlSchemaCompareValuesWhtsp	(xmlSchemaValPtr x,
						 xmlSchemaWhitespaceValueType xws,
						 xmlSchemaValPtr y,
						 xmlSchemaWhitespaceValueType yws);
XMLPUBFUN xmlSchemaValPtr XMLCALL
		xmlSchemaCopyValue		(xmlSchemaValPtr val);
XMLPUBFUN xmlSchemaValType XMLCALL
		xmlSchemaGetValType		(xmlSchemaValPtr val);

#ifdef __cplusplus
}
#endif

#endif /* LIBXML_SCHEMAS_ENABLED */
#endif /* __XML_SCHEMA_TYPES_H__ */
