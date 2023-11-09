/*
 * Summary: internals routines and limits exported by the parser.
 * Description: this module exports a number of internal parsing routines
 *              they are not really all intended for applications but
 *              can prove useful doing low level processing.
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */

#ifndef __XML_PARSER_INTERNALS_H__
#define __XML_PARSER_INTERNALS_H__

#include <libxml/xmlversion.h>
#include <libxml/parser.h>
#include <libxml/HTMLparser.h>
#include <libxml/chvalid.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xmlParserMaxDepth:
 *
 * arbitrary depth limit for the XML documents that we allow to
 * process. This is not a limitation of the parser but a safety
 * boundary feature, use XML_PARSE_HUGE option to override it.
 */
XMLPUBVAR unsigned int xmlParserMaxDepth;

/**
 * XML_MAX_TEXT_LENGTH:
 *
 * Maximum size allowed for a single text node when building a tree.
 * This is not a limitation of the parser but a safety boundary feature,
 * use XML_PARSE_HUGE option to override it.
 * Introduced in 2.9.0
 */
#define XML_MAX_TEXT_LENGTH 10000000

/**
 * XML_MAX_NAME_LENGTH:
 *
 * Maximum size allowed for a markup identifier.
 * This is not a limitation of the parser but a safety boundary feature,
 * use XML_PARSE_HUGE option to override it.
 * Note that with the use of parsing dictionaries overriding the limit
 * may result in more runtime memory usage in face of "unfriendly' content
 * Introduced in 2.9.0
 */
#define XML_MAX_NAME_LENGTH 50000

/**
 * XML_MAX_DICTIONARY_LIMIT:
 *
 * Maximum size allowed by the parser for a dictionary by default
 * This is not a limitation of the parser but a safety boundary feature,
 * use XML_PARSE_HUGE option to override it.
 * Introduced in 2.9.0
 */
#define XML_MAX_DICTIONARY_LIMIT 10000000

/**
 * XML_MAX_LOOKUP_LIMIT:
 *
 * Maximum size allowed by the parser for ahead lookup
 * This is an upper boundary enforced by the parser to avoid bad
 * behaviour on "unfriendly' content
 * Introduced in 2.9.0
 */
#define XML_MAX_LOOKUP_LIMIT 10000000

/**
 * XML_MAX_NAMELEN:
 *
 * Identifiers can be longer, but this will be more costly
 * at runtime.
 */
#define XML_MAX_NAMELEN 100

/**
 * INPUT_CHUNK:
 *
 * The parser tries to always have that amount of input ready.
 * One of the point is providing context when reporting errors.
 */
#define INPUT_CHUNK	250

/************************************************************************
 *									*
 * UNICODE version of the macros.					*
 *									*
 ************************************************************************/
/**
 * IS_BYTE_CHAR:
 * @c:  an byte value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 * [2] Char ::= #x9 | #xA | #xD | [#x20...]
 * any byte character in the accepted range
 */
#define IS_BYTE_CHAR(c)	 xmlIsChar_ch(c)

/**
 * IS_CHAR:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 * [2] Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
 *                  | [#x10000-#x10FFFF]
 * any Unicode character, excluding the surrogate blocks, FFFE, and FFFF.
 */
#define IS_CHAR(c)   xmlIsCharQ(c)

/**
 * IS_CHAR_CH:
 * @c: an xmlChar (usually an unsigned char)
 *
 * Behaves like IS_CHAR on single-byte value
 */
#define IS_CHAR_CH(c)  xmlIsChar_ch(c)

/**
 * IS_BLANK:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 * [3] S ::= (#x20 | #x9 | #xD | #xA)+
 */
#define IS_BLANK(c)  xmlIsBlankQ(c)

/**
 * IS_BLANK_CH:
 * @c:  an xmlChar value (normally unsigned char)
 *
 * Behaviour same as IS_BLANK
 */
#define IS_BLANK_CH(c)  xmlIsBlank_ch(c)

/**
 * IS_BASECHAR:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 * [85] BaseChar ::= ... long list see REC ...
 */
#define IS_BASECHAR(c) xmlIsBaseCharQ(c)

/**
 * IS_DIGIT:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 * [88] Digit ::= ... long list see REC ...
 */
#define IS_DIGIT(c) xmlIsDigitQ(c)

/**
 * IS_DIGIT_CH:
 * @c:  an xmlChar value (usually an unsigned char)
 *
 * Behaves like IS_DIGIT but with a single byte argument
 */
#define IS_DIGIT_CH(c)  xmlIsDigit_ch(c)

/**
 * IS_COMBINING:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 * [87] CombiningChar ::= ... long list see REC ...
 */
#define IS_COMBINING(c) xmlIsCombiningQ(c)

/**
 * IS_COMBINING_CH:
 * @c:  an xmlChar (usually an unsigned char)
 *
 * Always false (all combining chars > 0xff)
 */
#define IS_COMBINING_CH(c) 0

/**
 * IS_EXTENDER:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 *
 * [89] Extender ::= #x00B7 | #x02D0 | #x02D1 | #x0387 | #x0640 |
 *                   #x0E46 | #x0EC6 | #x3005 | [#x3031-#x3035] |
 *                   [#x309D-#x309E] | [#x30FC-#x30FE]
 */
#define IS_EXTENDER(c) xmlIsExtenderQ(c)

/**
 * IS_EXTENDER_CH:
 * @c:  an xmlChar value (usually an unsigned char)
 *
 * Behaves like IS_EXTENDER but with a single-byte argument
 */
#define IS_EXTENDER_CH(c)  xmlIsExtender_ch(c)

/**
 * IS_IDEOGRAPHIC:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 *
 * [86] Ideographic ::= [#x4E00-#x9FA5] | #x3007 | [#x3021-#x3029]
 */
#define IS_IDEOGRAPHIC(c) xmlIsIdeographicQ(c)

/**
 * IS_LETTER:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 *
 * [84] Letter ::= BaseChar | Ideographic
 */
#define IS_LETTER(c) (IS_BASECHAR(c) || IS_IDEOGRAPHIC(c))

/**
 * IS_LETTER_CH:
 * @c:  an xmlChar value (normally unsigned char)
 *
 * Macro behaves like IS_LETTER, but only check base chars
 *
 */
#define IS_LETTER_CH(c) xmlIsBaseChar_ch(c)

/**
 * IS_ASCII_LETTER:
 * @c: an xmlChar value
 *
 * Macro to check [a-zA-Z]
 *
 */
#define IS_ASCII_LETTER(c)	(((0x41 <= (c)) && ((c) <= 0x5a)) || \
				 ((0x61 <= (c)) && ((c) <= 0x7a)))

/**
 * IS_ASCII_DIGIT:
 * @c: an xmlChar value
 *
 * Macro to check [0-9]
 *
 */
#define IS_ASCII_DIGIT(c)	((0x30 <= (c)) && ((c) <= 0x39))

/**
 * IS_PUBIDCHAR:
 * @c:  an UNICODE value (int)
 *
 * Macro to check the following production in the XML spec:
 *
 *
 * [13] PubidChar ::= #x20 | #xD | #xA | [a-zA-Z0-9] | [-'()+,./:=?;!*#@$_%]
 */
#define IS_PUBIDCHAR(c)	xmlIsPubidCharQ(c)

/**
 * IS_PUBIDCHAR_CH:
 * @c:  an xmlChar value (normally unsigned char)
 *
 * Same as IS_PUBIDCHAR but for single-byte value
 */
#define IS_PUBIDCHAR_CH(c) xmlIsPubidChar_ch(c)

/**
 * SKIP_EOL:
 * @p:  and UTF8 string pointer
 *
 * Skips the end of line chars.
 */
#define SKIP_EOL(p)							\
    if (*(p) == 0x13) { p++ ; if (*(p) == 0x10) p++; }			\
    if (*(p) == 0x10) { p++ ; if (*(p) == 0x13) p++; }

/**
 * MOVETO_ENDTAG:
 * @p:  and UTF8 string pointer
 *
 * Skips to the next '>' char.
 */
#define MOVETO_ENDTAG(p)						\
    while ((*p) && (*(p) != '>')) (p)++

/**
 * MOVETO_STARTTAG:
 * @p:  and UTF8 string pointer
 *
 * Skips to the next '<' char.
 */
#define MOVETO_STARTTAG(p)						\
    while ((*p) && (*(p) != '<')) (p)++

/**
 * Global variables used for predefined strings.
 */
XMLPUBVAR const xmlChar xmlStringText[];
XMLPUBVAR const xmlChar xmlStringTextNoenc[];
XMLPUBVAR const xmlChar xmlStringComment[];

/*
 * Function to finish the work of the macros where needed.
 */
XMLPUBFUN int XMLCALL                   xmlIsLetter     (int c);

/**
 * Parser context.
 */
XMLPUBFUN xmlParserCtxtPtr XMLCALL
			xmlCreateFileParserCtxt	(const char *filename);
XMLPUBFUN xmlParserCtxtPtr XMLCALL
			xmlCreateURLParserCtxt	(const char *filename,
						 int options);
XMLPUBFUN xmlParserCtxtPtr XMLCALL
			xmlCreateMemoryParserCtxt(const char *buffer,
						 int size);
XMLPUBFUN xmlParserCtxtPtr XMLCALL
			xmlCreateEntityParserCtxt(const xmlChar *URL,
						 const xmlChar *ID,
						 const xmlChar *base);
XMLPUBFUN int XMLCALL
			xmlSwitchEncoding	(xmlParserCtxtPtr ctxt,
						 xmlCharEncoding enc);
XMLPUBFUN int XMLCALL
			xmlSwitchToEncoding	(xmlParserCtxtPtr ctxt,
					 xmlCharEncodingHandlerPtr handler);
XMLPUBFUN int XMLCALL
			xmlSwitchInputEncoding	(xmlParserCtxtPtr ctxt,
						 xmlParserInputPtr input,
					 xmlCharEncodingHandlerPtr handler);

#ifdef IN_LIBXML
/* internal error reporting */
XMLPUBFUN void XMLCALL
			__xmlErrEncoding	(xmlParserCtxtPtr ctxt,
						 xmlParserErrors xmlerr,
						 const char *msg,
						 const xmlChar * str1,
						 const xmlChar * str2) LIBXML_ATTR_FORMAT(3,0);
#endif

/**
 * Input Streams.
 */
XMLPUBFUN xmlParserInputPtr XMLCALL
			xmlNewStringInputStream	(xmlParserCtxtPtr ctxt,
						 const xmlChar *buffer);
XMLPUBFUN xmlParserInputPtr XMLCALL
			xmlNewEntityInputStream	(xmlParserCtxtPtr ctxt,
						 xmlEntityPtr entity);
XMLPUBFUN int XMLCALL
			xmlPushInput		(xmlParserCtxtPtr ctxt,
						 xmlParserInputPtr input);
XMLPUBFUN xmlChar XMLCALL
			xmlPopInput		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlFreeInputStream	(xmlParserInputPtr input);
XMLPUBFUN xmlParserInputPtr XMLCALL
			xmlNewInputFromFile	(xmlParserCtxtPtr ctxt,
						 const char *filename);
XMLPUBFUN xmlParserInputPtr XMLCALL
			xmlNewInputStream	(xmlParserCtxtPtr ctxt);

/**
 * Namespaces.
 */
XMLPUBFUN xmlChar * XMLCALL
			xmlSplitQName		(xmlParserCtxtPtr ctxt,
						 const xmlChar *name,
						 xmlChar **prefix);

/**
 * Generic production rules.
 */
XMLPUBFUN const xmlChar * XMLCALL
			xmlParseName		(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseNmtoken		(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseEntityValue	(xmlParserCtxtPtr ctxt,
						 xmlChar **orig);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseAttValue	(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseSystemLiteral	(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParsePubidLiteral	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseCharData	(xmlParserCtxtPtr ctxt,
						 int cdata);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseExternalID	(xmlParserCtxtPtr ctxt,
						 xmlChar **publicID,
						 int strict);
XMLPUBFUN void XMLCALL
			xmlParseComment		(xmlParserCtxtPtr ctxt);
XMLPUBFUN const xmlChar * XMLCALL
			xmlParsePITarget	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParsePI		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseNotationDecl	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseEntityDecl	(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL
			xmlParseDefaultDecl	(xmlParserCtxtPtr ctxt,
						 xmlChar **value);
XMLPUBFUN xmlEnumerationPtr XMLCALL
			xmlParseNotationType	(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlEnumerationPtr XMLCALL
			xmlParseEnumerationType	(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL
			xmlParseEnumeratedType	(xmlParserCtxtPtr ctxt,
						 xmlEnumerationPtr *tree);
XMLPUBFUN int XMLCALL
			xmlParseAttributeType	(xmlParserCtxtPtr ctxt,
						 xmlEnumerationPtr *tree);
XMLPUBFUN void XMLCALL
			xmlParseAttributeListDecl(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlElementContentPtr XMLCALL
			xmlParseElementMixedContentDecl
						(xmlParserCtxtPtr ctxt,
						 int inputchk);
XMLPUBFUN xmlElementContentPtr XMLCALL
			xmlParseElementChildrenContentDecl
						(xmlParserCtxtPtr ctxt,
						 int inputchk);
XMLPUBFUN int XMLCALL
			xmlParseElementContentDecl(xmlParserCtxtPtr ctxt,
						 const xmlChar *name,
						 xmlElementContentPtr *result);
XMLPUBFUN int XMLCALL
			xmlParseElementDecl	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseMarkupDecl	(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL
			xmlParseCharRef		(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlEntityPtr XMLCALL
			xmlParseEntityRef	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseReference	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParsePEReference	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseDocTypeDecl	(xmlParserCtxtPtr ctxt);
#ifdef LIBXML_SAX1_ENABLED
XMLPUBFUN const xmlChar * XMLCALL
			xmlParseAttribute	(xmlParserCtxtPtr ctxt,
						 xmlChar **value);
XMLPUBFUN const xmlChar * XMLCALL
			xmlParseStartTag	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseEndTag		(xmlParserCtxtPtr ctxt);
#endif /* LIBXML_SAX1_ENABLED */
XMLPUBFUN void XMLCALL
			xmlParseCDSect		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseContent		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseElement		(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseVersionNum	(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseVersionInfo	(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlParseEncName		(xmlParserCtxtPtr ctxt);
XMLPUBFUN const xmlChar * XMLCALL
			xmlParseEncodingDecl	(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL
			xmlParseSDDecl		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseXMLDecl		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseTextDecl	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseMisc		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
			xmlParseExternalSubset	(xmlParserCtxtPtr ctxt,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID);
/**
 * XML_SUBSTITUTE_NONE:
 *
 * If no entities need to be substituted.
 */
#define XML_SUBSTITUTE_NONE	0
/**
 * XML_SUBSTITUTE_REF:
 *
 * Whether general entities need to be substituted.
 */
#define XML_SUBSTITUTE_REF	1
/**
 * XML_SUBSTITUTE_PEREF:
 *
 * Whether parameter entities need to be substituted.
 */
#define XML_SUBSTITUTE_PEREF	2
/**
 * XML_SUBSTITUTE_BOTH:
 *
 * Both general and parameter entities need to be substituted.
 */
#define XML_SUBSTITUTE_BOTH	3

XMLPUBFUN xmlChar * XMLCALL
		xmlStringDecodeEntities		(xmlParserCtxtPtr ctxt,
						 const xmlChar *str,
						 int what,
						 xmlChar end,
						 xmlChar  end2,
						 xmlChar end3);
XMLPUBFUN xmlChar * XMLCALL
		xmlStringLenDecodeEntities	(xmlParserCtxtPtr ctxt,
						 const xmlChar *str,
						 int len,
						 int what,
						 xmlChar end,
						 xmlChar  end2,
						 xmlChar end3);

/*
 * Generated by MACROS on top of parser.c c.f. PUSH_AND_POP.
 */
XMLPUBFUN int XMLCALL			nodePush		(xmlParserCtxtPtr ctxt,
						 xmlNodePtr value);
XMLPUBFUN xmlNodePtr XMLCALL		nodePop			(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL			inputPush		(xmlParserCtxtPtr ctxt,
						 xmlParserInputPtr value);
XMLPUBFUN xmlParserInputPtr XMLCALL	inputPop		(xmlParserCtxtPtr ctxt);
XMLPUBFUN const xmlChar * XMLCALL	namePop			(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL			namePush		(xmlParserCtxtPtr ctxt,
						 const xmlChar *value);

/*
 * other commodities shared between parser.c and parserInternals.
 */
XMLPUBFUN int XMLCALL			xmlSkipBlankChars	(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL			xmlStringCurrentChar	(xmlParserCtxtPtr ctxt,
						 const xmlChar *cur,
						 int *len);
XMLPUBFUN void XMLCALL			xmlParserHandlePEReference(xmlParserCtxtPtr ctxt);
XMLPUBFUN int XMLCALL			xmlCheckLanguageID	(const xmlChar *lang);

/*
 * Really core function shared with HTML parser.
 */
XMLPUBFUN int XMLCALL			xmlCurrentChar		(xmlParserCtxtPtr ctxt,
						 int *len);
XMLPUBFUN int XMLCALL		xmlCopyCharMultiByte	(xmlChar *out,
						 int val);
XMLPUBFUN int XMLCALL			xmlCopyChar		(int len,
						 xmlChar *out,
						 int val);
XMLPUBFUN void XMLCALL			xmlNextChar		(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL			xmlParserInputShrink	(xmlParserInputPtr in);

#ifdef LIBXML_HTML_ENABLED
/*
 * Actually comes from the HTML parser but launched from the init stuff.
 */
XMLPUBFUN void XMLCALL			htmlInitAutoClose	(void);
XMLPUBFUN htmlParserCtxtPtr XMLCALL	htmlCreateFileParserCtxt(const char *filename,
	                                         const char *encoding);
#endif

/*
 * Specific function to keep track of entities references
 * and used by the XSLT debugger.
 */
#ifdef LIBXML_LEGACY_ENABLED
/**
 * xmlEntityReferenceFunc:
 * @ent: the entity
 * @firstNode:  the fist node in the chunk
 * @lastNode:  the last nod in the chunk
 *
 * Callback function used when one needs to be able to track back the
 * provenance of a chunk of nodes inherited from an entity replacement.
 */
typedef	void	(*xmlEntityReferenceFunc)	(xmlEntityPtr ent,
						 xmlNodePtr firstNode,
						 xmlNodePtr lastNode);

XMLPUBFUN void XMLCALL		xmlSetEntityReferenceFunc	(xmlEntityReferenceFunc func);

XMLPUBFUN xmlChar * XMLCALL
			xmlParseQuotedString	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
                        xmlParseNamespace       (xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlNamespaceParseNSDef	(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlScanName		(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlNamespaceParseNCName	(xmlParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL	xmlParserHandleReference(xmlParserCtxtPtr ctxt);
XMLPUBFUN xmlChar * XMLCALL
			xmlNamespaceParseQName	(xmlParserCtxtPtr ctxt,
						 xmlChar **prefix);
/**
 * Entities
 */
XMLPUBFUN xmlChar * XMLCALL
		xmlDecodeEntities		(xmlParserCtxtPtr ctxt,
						 int len,
						 int what,
						 xmlChar end,
						 xmlChar  end2,
						 xmlChar end3);
XMLPUBFUN void XMLCALL
			xmlHandleEntity		(xmlParserCtxtPtr ctxt,
						 xmlEntityPtr entity);

#endif /* LIBXML_LEGACY_ENABLED */

#ifdef IN_LIBXML
/*
 * internal only
 */
XMLPUBFUN void XMLCALL
	xmlErrMemory		(xmlParserCtxtPtr ctxt,
				 const char *extra);
#endif

#ifdef __cplusplus
}
#endif
#endif /* __XML_PARSER_INTERNALS_H__ */
