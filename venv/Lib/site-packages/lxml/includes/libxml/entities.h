/*
 * Summary: interface for the XML entities handling
 * Description: this module provides some of the entity API needed
 *              for the parser and applications.
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */

#ifndef __XML_ENTITIES_H__
#define __XML_ENTITIES_H__

#include <libxml/xmlversion.h>
#include <libxml/tree.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The different valid entity types.
 */
typedef enum {
    XML_INTERNAL_GENERAL_ENTITY = 1,
    XML_EXTERNAL_GENERAL_PARSED_ENTITY = 2,
    XML_EXTERNAL_GENERAL_UNPARSED_ENTITY = 3,
    XML_INTERNAL_PARAMETER_ENTITY = 4,
    XML_EXTERNAL_PARAMETER_ENTITY = 5,
    XML_INTERNAL_PREDEFINED_ENTITY = 6
} xmlEntityType;

/*
 * An unit of storage for an entity, contains the string, the value
 * and the linkind data needed for the linking in the hash table.
 */

struct _xmlEntity {
    void           *_private;	        /* application data */
    xmlElementType          type;       /* XML_ENTITY_DECL, must be second ! */
    const xmlChar          *name;	/* Entity name */
    struct _xmlNode    *children;	/* First child link */
    struct _xmlNode        *last;	/* Last child link */
    struct _xmlDtd       *parent;	/* -> DTD */
    struct _xmlNode        *next;	/* next sibling link  */
    struct _xmlNode        *prev;	/* previous sibling link  */
    struct _xmlDoc          *doc;       /* the containing document */

    xmlChar                *orig;	/* content without ref substitution */
    xmlChar             *content;	/* content or ndata if unparsed */
    int                   length;	/* the content length */
    xmlEntityType          etype;	/* The entity type */
    const xmlChar    *ExternalID;	/* External identifier for PUBLIC */
    const xmlChar      *SystemID;	/* URI for a SYSTEM or PUBLIC Entity */

    struct _xmlEntity     *nexte;	/* unused */
    const xmlChar           *URI;	/* the full URI as computed */
    int                    owner;	/* does the entity own the childrens */
    int			 checked;	/* was the entity content checked */
					/* this is also used to count entities
					 * references done from that entity
					 * and if it contains '<' */
};

/*
 * All entities are stored in an hash table.
 * There is 2 separate hash tables for global and parameter entities.
 */

typedef struct _xmlHashTable xmlEntitiesTable;
typedef xmlEntitiesTable *xmlEntitiesTablePtr;

/*
 * External functions:
 */

#ifdef LIBXML_LEGACY_ENABLED
XMLPUBFUN void XMLCALL
		xmlInitializePredefinedEntities	(void);
#endif /* LIBXML_LEGACY_ENABLED */

XMLPUBFUN xmlEntityPtr XMLCALL
			xmlNewEntity		(xmlDocPtr doc,
						 const xmlChar *name,
						 int type,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID,
						 const xmlChar *content);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlAddDocEntity		(xmlDocPtr doc,
						 const xmlChar *name,
						 int type,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID,
						 const xmlChar *content);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlAddDtdEntity		(xmlDocPtr doc,
						 const xmlChar *name,
						 int type,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID,
						 const xmlChar *content);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlGetPredefinedEntity	(const xmlChar *name);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlGetDocEntity		(const xmlDoc *doc,
						 const xmlChar *name);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlGetDtdEntity		(xmlDocPtr doc,
						 const xmlChar *name);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlGetParameterEntity	(xmlDocPtr doc,
						 const xmlChar *name);
#ifdef LIBXML_LEGACY_ENABLED
XMLPUBFUN const xmlChar * XMLCALL
			xmlEncodeEntities	(xmlDocPtr doc,
						 const xmlChar *input);
#endif /* LIBXML_LEGACY_ENABLED */
XMLPUBFUN xmlChar * XMLCALL
			xmlEncodeEntitiesReentrant(xmlDocPtr doc,
						 const xmlChar *input);
XMLPUBFUN xmlChar * XMLCALL
			xmlEncodeSpecialChars	(const xmlDoc *doc,
						 const xmlChar *input);
XMLPUBFUN xmlEntitiesTablePtr XMLCALL
			xmlCreateEntitiesTable	(void);
#ifdef LIBXML_TREE_ENABLED
XMLPUBFUN xmlEntitiesTablePtr XMLCALL
			xmlCopyEntitiesTable	(xmlEntitiesTablePtr table);
#endif /* LIBXML_TREE_ENABLED */
XMLPUBFUN void XMLCALL
			xmlFreeEntitiesTable	(xmlEntitiesTablePtr table);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN void XMLCALL
			xmlDumpEntitiesTable	(xmlBufferPtr buf,
						 xmlEntitiesTablePtr table);
XMLPUBFUN void XMLCALL
			xmlDumpEntityDecl	(xmlBufferPtr buf,
						 xmlEntityPtr ent);
#endif /* LIBXML_OUTPUT_ENABLED */
#ifdef LIBXML_LEGACY_ENABLED
XMLPUBFUN void XMLCALL
			xmlCleanupPredefinedEntities(void);
#endif /* LIBXML_LEGACY_ENABLED */


#ifdef __cplusplus
}
#endif

# endif /* __XML_ENTITIES_H__ */
