/**
 * Summary: interfaces to the Catalog handling system
 * Description: the catalog module implements the support for
 * XML Catalogs and SGML catalogs
 *
 * SGML Open Technical Resolution TR9401:1997.
 * http://www.jclark.com/sp/catalog.htm
 *
 * XML Catalogs Working Draft 06 August 2001
 * http://www.oasis-open.org/committees/entity/spec-2001-08-06.html
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */

#ifndef __XML_CATALOG_H__
#define __XML_CATALOG_H__

#include <stdio.h>

#include <libxml/xmlversion.h>
#include <libxml/xmlstring.h>
#include <libxml/tree.h>

#ifdef LIBXML_CATALOG_ENABLED

#ifdef __cplusplus
extern "C" {
#endif

/**
 * XML_CATALOGS_NAMESPACE:
 *
 * The namespace for the XML Catalogs elements.
 */
#define XML_CATALOGS_NAMESPACE					\
    (const xmlChar *) "urn:oasis:names:tc:entity:xmlns:xml:catalog"
/**
 * XML_CATALOG_PI:
 *
 * The specific XML Catalog Processing Instruction name.
 */
#define XML_CATALOG_PI						\
    (const xmlChar *) "oasis-xml-catalog"

/*
 * The API is voluntarily limited to general cataloging.
 */
typedef enum {
    XML_CATA_PREFER_NONE = 0,
    XML_CATA_PREFER_PUBLIC = 1,
    XML_CATA_PREFER_SYSTEM
} xmlCatalogPrefer;

typedef enum {
    XML_CATA_ALLOW_NONE = 0,
    XML_CATA_ALLOW_GLOBAL = 1,
    XML_CATA_ALLOW_DOCUMENT = 2,
    XML_CATA_ALLOW_ALL = 3
} xmlCatalogAllow;

typedef struct _xmlCatalog xmlCatalog;
typedef xmlCatalog *xmlCatalogPtr;

/*
 * Operations on a given catalog.
 */
XMLPUBFUN xmlCatalogPtr XMLCALL
		xmlNewCatalog		(int sgml);
XMLPUBFUN xmlCatalogPtr XMLCALL
		xmlLoadACatalog		(const char *filename);
XMLPUBFUN xmlCatalogPtr XMLCALL
		xmlLoadSGMLSuperCatalog	(const char *filename);
XMLPUBFUN int XMLCALL
		xmlConvertSGMLCatalog	(xmlCatalogPtr catal);
XMLPUBFUN int XMLCALL
		xmlACatalogAdd		(xmlCatalogPtr catal,
					 const xmlChar *type,
					 const xmlChar *orig,
					 const xmlChar *replace);
XMLPUBFUN int XMLCALL
		xmlACatalogRemove	(xmlCatalogPtr catal,
					 const xmlChar *value);
XMLPUBFUN xmlChar * XMLCALL
		xmlACatalogResolve	(xmlCatalogPtr catal,
					 const xmlChar *pubID,
	                                 const xmlChar *sysID);
XMLPUBFUN xmlChar * XMLCALL
		xmlACatalogResolveSystem(xmlCatalogPtr catal,
					 const xmlChar *sysID);
XMLPUBFUN xmlChar * XMLCALL
		xmlACatalogResolvePublic(xmlCatalogPtr catal,
					 const xmlChar *pubID);
XMLPUBFUN xmlChar * XMLCALL
		xmlACatalogResolveURI	(xmlCatalogPtr catal,
					 const xmlChar *URI);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN void XMLCALL
		xmlACatalogDump		(xmlCatalogPtr catal,
					 FILE *out);
#endif /* LIBXML_OUTPUT_ENABLED */
XMLPUBFUN void XMLCALL
		xmlFreeCatalog		(xmlCatalogPtr catal);
XMLPUBFUN int XMLCALL
		xmlCatalogIsEmpty	(xmlCatalogPtr catal);

/*
 * Global operations.
 */
XMLPUBFUN void XMLCALL
		xmlInitializeCatalog	(void);
XMLPUBFUN int XMLCALL
		xmlLoadCatalog		(const char *filename);
XMLPUBFUN void XMLCALL
		xmlLoadCatalogs		(const char *paths);
XMLPUBFUN void XMLCALL
		xmlCatalogCleanup	(void);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN void XMLCALL
		xmlCatalogDump		(FILE *out);
#endif /* LIBXML_OUTPUT_ENABLED */
XMLPUBFUN xmlChar * XMLCALL
		xmlCatalogResolve	(const xmlChar *pubID,
	                                 const xmlChar *sysID);
XMLPUBFUN xmlChar * XMLCALL
		xmlCatalogResolveSystem	(const xmlChar *sysID);
XMLPUBFUN xmlChar * XMLCALL
		xmlCatalogResolvePublic	(const xmlChar *pubID);
XMLPUBFUN xmlChar * XMLCALL
		xmlCatalogResolveURI	(const xmlChar *URI);
XMLPUBFUN int XMLCALL
		xmlCatalogAdd		(const xmlChar *type,
					 const xmlChar *orig,
					 const xmlChar *replace);
XMLPUBFUN int XMLCALL
		xmlCatalogRemove	(const xmlChar *value);
XMLPUBFUN xmlDocPtr XMLCALL
		xmlParseCatalogFile	(const char *filename);
XMLPUBFUN int XMLCALL
		xmlCatalogConvert	(void);

/*
 * Strictly minimal interfaces for per-document catalogs used
 * by the parser.
 */
XMLPUBFUN void XMLCALL
		xmlCatalogFreeLocal	(void *catalogs);
XMLPUBFUN void * XMLCALL
		xmlCatalogAddLocal	(void *catalogs,
					 const xmlChar *URL);
XMLPUBFUN xmlChar * XMLCALL
		xmlCatalogLocalResolve	(void *catalogs,
					 const xmlChar *pubID,
	                                 const xmlChar *sysID);
XMLPUBFUN xmlChar * XMLCALL
		xmlCatalogLocalResolveURI(void *catalogs,
					 const xmlChar *URI);
/*
 * Preference settings.
 */
XMLPUBFUN int XMLCALL
		xmlCatalogSetDebug	(int level);
XMLPUBFUN xmlCatalogPrefer XMLCALL
		xmlCatalogSetDefaultPrefer(xmlCatalogPrefer prefer);
XMLPUBFUN void XMLCALL
		xmlCatalogSetDefaults	(xmlCatalogAllow allow);
XMLPUBFUN xmlCatalogAllow XMLCALL
		xmlCatalogGetDefaults	(void);


/* DEPRECATED interfaces */
XMLPUBFUN const xmlChar * XMLCALL
		xmlCatalogGetSystem	(const xmlChar *sysID);
XMLPUBFUN const xmlChar * XMLCALL
		xmlCatalogGetPublic	(const xmlChar *pubID);

#ifdef __cplusplus
}
#endif
#endif /* LIBXML_CATALOG_ENABLED */
#endif /* __XML_CATALOG_H__ */
