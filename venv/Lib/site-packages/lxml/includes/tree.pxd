from libc cimport stdio
from libc.string cimport const_char, const_uchar

cdef extern from "lxml-version.h":
    # deprecated declaration, use etreepublic.pxd instead
    cdef char* LXML_VERSION_STRING

cdef extern from "libxml/xmlversion.h":
    cdef const_char* xmlParserVersion
    cdef int LIBXML_VERSION

cdef extern from "libxml/xmlstring.h":
    ctypedef unsigned char xmlChar
    ctypedef const xmlChar const_xmlChar "const xmlChar"
    cdef int xmlStrlen(const_xmlChar* str) nogil
    cdef xmlChar* xmlStrdup(const_xmlChar* cur) nogil
    cdef int xmlStrncmp(const_xmlChar* str1, const_xmlChar* str2, int length) nogil
    cdef int xmlStrcmp(const_xmlChar* str1, const_xmlChar* str2) nogil
    cdef int xmlStrcasecmp(const xmlChar *str1, const xmlChar *str2) nogil
    cdef const_xmlChar* xmlStrstr(const_xmlChar* str1, const_xmlChar* str2) nogil
    cdef const_xmlChar* xmlStrchr(const_xmlChar* str1, xmlChar ch) nogil
    cdef const_xmlChar* _xcstr "(const xmlChar*)PyBytes_AS_STRING" (object s)

cdef extern from "libxml/encoding.h":
    ctypedef enum xmlCharEncoding:
        XML_CHAR_ENCODING_ERROR = -1 # No char encoding detected
        XML_CHAR_ENCODING_NONE = 0 # No char encoding detected
        XML_CHAR_ENCODING_UTF8 = 1 # UTF-8
        XML_CHAR_ENCODING_UTF16LE = 2 # UTF-16 little endian
        XML_CHAR_ENCODING_UTF16BE = 3 # UTF-16 big endian
        XML_CHAR_ENCODING_UCS4LE = 4 # UCS-4 little endian
        XML_CHAR_ENCODING_UCS4BE = 5 # UCS-4 big endian
        XML_CHAR_ENCODING_EBCDIC = 6 # EBCDIC uh!
        XML_CHAR_ENCODING_UCS4_2143 = 7 # UCS-4 unusual ordering
        XML_CHAR_ENCODING_UCS4_3412 = 8 # UCS-4 unusual ordering
        XML_CHAR_ENCODING_UCS2 = 9 # UCS-2
        XML_CHAR_ENCODING_8859_1 = 10 # ISO-8859-1 ISO Latin 1
        XML_CHAR_ENCODING_8859_2 = 11 # ISO-8859-2 ISO Latin 2
        XML_CHAR_ENCODING_8859_3 = 12 # ISO-8859-3
        XML_CHAR_ENCODING_8859_4 = 13 # ISO-8859-4
        XML_CHAR_ENCODING_8859_5 = 14 # ISO-8859-5
        XML_CHAR_ENCODING_8859_6 = 15 # ISO-8859-6
        XML_CHAR_ENCODING_8859_7 = 16 # ISO-8859-7
        XML_CHAR_ENCODING_8859_8 = 17 # ISO-8859-8
        XML_CHAR_ENCODING_8859_9 = 18 # ISO-8859-9
        XML_CHAR_ENCODING_2022_JP = 19 # ISO-2022-JP
        XML_CHAR_ENCODING_SHIFT_JIS = 20 # Shift_JIS
        XML_CHAR_ENCODING_EUC_JP = 21 # EUC-JP
        XML_CHAR_ENCODING_ASCII = 22 # pure ASCII

    ctypedef struct xmlCharEncodingHandler
    cdef xmlCharEncodingHandler* xmlFindCharEncodingHandler(char* name) nogil
    cdef xmlCharEncodingHandler* xmlGetCharEncodingHandler(
        xmlCharEncoding enc) nogil
    cdef int xmlCharEncCloseFunc(xmlCharEncodingHandler* handler) nogil
    cdef xmlCharEncoding xmlDetectCharEncoding(const_xmlChar* text, int len) nogil
    cdef const_char* xmlGetCharEncodingName(xmlCharEncoding enc) nogil
    cdef xmlCharEncoding xmlParseCharEncoding(char* name) nogil
    ctypedef int (*xmlCharEncodingOutputFunc)(
            unsigned char *out_buf, int *outlen, const_uchar *in_buf, int *inlen)

cdef extern from "libxml/chvalid.h":
    cdef int xmlIsChar_ch(char c) nogil
    cdef int xmlIsCharQ(int ch) nogil

cdef extern from "libxml/hash.h":
    ctypedef struct xmlHashTable
    ctypedef void (*xmlHashScanner)(void* payload, void* data, const_xmlChar* name) # may require GIL!
    void xmlHashScan(xmlHashTable* table, xmlHashScanner f, void* data) nogil
    void* xmlHashLookup(xmlHashTable* table, const_xmlChar* name) nogil
    ctypedef void (*xmlHashDeallocator)(void *payload, xmlChar *name)
    cdef xmlHashTable* xmlHashCreate(int size)
    cdef xmlHashTable* xmlHashCreateDict(int size, xmlDict *dict)
    cdef int xmlHashSize(xmlHashTable* table)
    cdef void xmlHashFree(xmlHashTable* table, xmlHashDeallocator f)

cdef extern from *: # actually "libxml/dict.h"
    # libxml/dict.h appears to be broken to include in C
    ctypedef struct xmlDict
    cdef const_xmlChar* xmlDictLookup(xmlDict* dict, const_xmlChar* name, int len) nogil
    cdef const_xmlChar* xmlDictExists(xmlDict* dict, const_xmlChar* name, int len) nogil
    cdef int xmlDictOwns(xmlDict* dict, const_xmlChar* name) nogil
    cdef size_t xmlDictSize(xmlDict* dict) nogil

cdef extern from "libxml/tree.h":
    ctypedef struct xmlDoc
    ctypedef struct xmlAttr
    ctypedef struct xmlNotationTable

    ctypedef enum xmlElementType:
        XML_ELEMENT_NODE=           1
        XML_ATTRIBUTE_NODE=         2
        XML_TEXT_NODE=              3
        XML_CDATA_SECTION_NODE=     4
        XML_ENTITY_REF_NODE=        5
        XML_ENTITY_NODE=            6
        XML_PI_NODE=                7
        XML_COMMENT_NODE=           8
        XML_DOCUMENT_NODE=          9
        XML_DOCUMENT_TYPE_NODE=     10
        XML_DOCUMENT_FRAG_NODE=     11
        XML_NOTATION_NODE=          12
        XML_HTML_DOCUMENT_NODE=     13
        XML_DTD_NODE=               14
        XML_ELEMENT_DECL=           15
        XML_ATTRIBUTE_DECL=         16
        XML_ENTITY_DECL=            17
        XML_NAMESPACE_DECL=         18
        XML_XINCLUDE_START=         19
        XML_XINCLUDE_END=           20

    ctypedef enum xmlElementTypeVal:
        XML_ELEMENT_TYPE_UNDEFINED= 0
        XML_ELEMENT_TYPE_EMPTY=     1
        XML_ELEMENT_TYPE_ANY=       2
        XML_ELEMENT_TYPE_MIXED=     3
        XML_ELEMENT_TYPE_ELEMENT=   4

    ctypedef enum xmlElementContentType:
        XML_ELEMENT_CONTENT_PCDATA=  1
        XML_ELEMENT_CONTENT_ELEMENT= 2
        XML_ELEMENT_CONTENT_SEQ=     3
        XML_ELEMENT_CONTENT_OR=      4

    ctypedef enum xmlElementContentOccur:
        XML_ELEMENT_CONTENT_ONCE= 1
        XML_ELEMENT_CONTENT_OPT=  2
        XML_ELEMENT_CONTENT_MULT= 3
        XML_ELEMENT_CONTENT_PLUS= 4

    ctypedef enum xmlAttributeType:
        XML_ATTRIBUTE_CDATA =      1
        XML_ATTRIBUTE_ID=          2
        XML_ATTRIBUTE_IDREF=       3
        XML_ATTRIBUTE_IDREFS=      4
        XML_ATTRIBUTE_ENTITY=      5
        XML_ATTRIBUTE_ENTITIES=    6
        XML_ATTRIBUTE_NMTOKEN=     7
        XML_ATTRIBUTE_NMTOKENS=    8
        XML_ATTRIBUTE_ENUMERATION= 9
        XML_ATTRIBUTE_NOTATION=    10
    
    ctypedef enum xmlAttributeDefault:
        XML_ATTRIBUTE_NONE=     1
        XML_ATTRIBUTE_REQUIRED= 2
        XML_ATTRIBUTE_IMPLIED=  3
        XML_ATTRIBUTE_FIXED=    4

    ctypedef enum xmlEntityType:
        XML_INTERNAL_GENERAL_ENTITY=          1
        XML_EXTERNAL_GENERAL_PARSED_ENTITY=   2
        XML_EXTERNAL_GENERAL_UNPARSED_ENTITY= 3
        XML_INTERNAL_PARAMETER_ENTITY=        4
        XML_EXTERNAL_PARAMETER_ENTITY=        5
        XML_INTERNAL_PREDEFINED_ENTITY=       6

    ctypedef struct xmlNs:
        const_xmlChar* href
        const_xmlChar* prefix
        xmlNs* next

    ctypedef struct xmlNode:
        void* _private
        xmlElementType   type
        const_xmlChar* name
        xmlNode* children
        xmlNode* last
        xmlNode* parent
        xmlNode* next
        xmlNode* prev
        xmlDoc* doc
        xmlChar* content
        xmlAttr* properties
        xmlNs* ns
        xmlNs* nsDef
        unsigned short line

    ctypedef struct xmlElementContent:
        xmlElementContentType type
        xmlElementContentOccur ocur
        const_xmlChar *name
        xmlElementContent *c1
        xmlElementContent *c2
        xmlElementContent *parent
        const_xmlChar *prefix

    ctypedef struct xmlEnumeration:
        xmlEnumeration *next
        const_xmlChar *name

    ctypedef struct xmlAttribute:
        void* _private
        xmlElementType type
        const_xmlChar* name
        xmlNode* children
        xmlNode* last
        xmlDtd* parent
        xmlNode* next
        xmlNode* prev
        xmlDoc* doc
        xmlAttribute* nexth
        xmlAttributeType atype
        xmlAttributeDefault def_ "def"
        const_xmlChar* defaultValue
        xmlEnumeration* tree
        const_xmlChar* prefix
        const_xmlChar* elem

    ctypedef struct xmlElement:
        void* _private
        xmlElementType   type
        const_xmlChar* name
        xmlNode* children
        xmlNode* last
        xmlNode* parent
        xmlNode* next
        xmlNode* prev
        xmlDoc* doc
        xmlElementTypeVal etype
        xmlElementContent* content
        xmlAttribute* attributes
        const_xmlChar* prefix
        void *contModel

    ctypedef struct xmlEntity:
        void* _private
        xmlElementType type
        const_xmlChar* name
        xmlNode* children
        xmlNode* last
        xmlDtd* parent
        xmlNode* next
        xmlNode* prev
        xmlDoc* doc
        xmlChar* orig
        xmlChar* content
        int length
        xmlEntityType etype
        const_xmlChar* ExternalID
        const_xmlChar* SystemID
        xmlEntity* nexte
        const_xmlChar* URI
        int owner
        int checked

    ctypedef struct xmlDtd:
        const_xmlChar* name
        const_xmlChar* ExternalID
        const_xmlChar* SystemID
        void* notations
        void* entities
        void* pentities
        void* attributes
        void* elements
        xmlNode* children
        xmlNode* last
        xmlDoc* doc

    ctypedef struct xmlDoc:
        xmlElementType type
        char* name
        xmlNode* children
        xmlNode* last
        xmlNode* parent
        xmlNode* next
        xmlNode* prev
        xmlDoc* doc
        xmlDict* dict
        xmlHashTable* ids
        int standalone
        const_xmlChar* version
        const_xmlChar* encoding
        const_xmlChar* URL
        void* _private
        xmlDtd* intSubset
        xmlDtd* extSubset
        
    ctypedef struct xmlAttr:
        void* _private
        xmlElementType type
        const_xmlChar* name
        xmlNode* children
        xmlNode* last
        xmlNode* parent
        xmlAttr* next
        xmlAttr* prev
        xmlDoc* doc
        xmlNs* ns
        xmlAttributeType atype

    ctypedef struct xmlID:
        const_xmlChar* value
        const_xmlChar* name
        xmlAttr* attr
        xmlDoc* doc
        
    ctypedef struct xmlBuffer

    ctypedef struct xmlBuf   # new in libxml2 2.9

    ctypedef struct xmlOutputBuffer:
        xmlBuf* buffer
        xmlBuf* conv
        int error

    const_xmlChar* XML_XML_NAMESPACE
        
    cdef void xmlFreeDoc(xmlDoc* cur) nogil
    cdef void xmlFreeDtd(xmlDtd* cur) nogil
    cdef void xmlFreeNode(xmlNode* cur) nogil
    cdef void xmlFreeNsList(xmlNs* ns) nogil
    cdef void xmlFreeNs(xmlNs* ns) nogil
    cdef void xmlFree(void* buf) nogil
    
    cdef xmlNode* xmlNewNode(xmlNs* ns, const_xmlChar* name) nogil
    cdef xmlNode* xmlNewDocText(xmlDoc* doc, const_xmlChar* content) nogil
    cdef xmlNode* xmlNewDocComment(xmlDoc* doc, const_xmlChar* content) nogil
    cdef xmlNode* xmlNewDocPI(xmlDoc* doc, const_xmlChar* name, const_xmlChar* content) nogil
    cdef xmlNode* xmlNewReference(xmlDoc* doc, const_xmlChar* name) nogil
    cdef xmlNode* xmlNewCDataBlock(xmlDoc* doc, const_xmlChar* text, int len) nogil
    cdef xmlNs* xmlNewNs(xmlNode* node, const_xmlChar* href, const_xmlChar* prefix) nogil
    cdef xmlNode* xmlAddChild(xmlNode* parent, xmlNode* cur) nogil
    cdef xmlNode* xmlReplaceNode(xmlNode* old, xmlNode* cur) nogil
    cdef xmlNode* xmlAddPrevSibling(xmlNode* cur, xmlNode* elem) nogil
    cdef xmlNode* xmlAddNextSibling(xmlNode* cur, xmlNode* elem) nogil
    cdef xmlNode* xmlNewDocNode(xmlDoc* doc, xmlNs* ns,
                                const_xmlChar* name, const_xmlChar* content) nogil
    cdef xmlDoc* xmlNewDoc(const_xmlChar* version) nogil
    cdef xmlAttr* xmlNewProp(xmlNode* node, const_xmlChar* name, const_xmlChar* value) nogil
    cdef xmlAttr* xmlNewNsProp(xmlNode* node, xmlNs* ns,
                               const_xmlChar* name, const_xmlChar* value) nogil
    cdef xmlChar* xmlGetNoNsProp(xmlNode* node, const_xmlChar* name) nogil
    cdef xmlChar* xmlGetNsProp(xmlNode* node, const_xmlChar* name, const_xmlChar* nameSpace) nogil
    cdef void xmlSetNs(xmlNode* node, xmlNs* ns) nogil
    cdef xmlAttr* xmlSetProp(xmlNode* node, const_xmlChar* name, const_xmlChar* value) nogil
    cdef xmlAttr* xmlSetNsProp(xmlNode* node, xmlNs* ns,
                               const_xmlChar* name, const_xmlChar* value) nogil
    cdef int xmlRemoveID(xmlDoc* doc, xmlAttr* cur) nogil
    cdef int xmlRemoveProp(xmlAttr* cur) nogil
    cdef void xmlFreePropList(xmlAttr* cur) nogil
    cdef xmlChar* xmlGetNodePath(xmlNode* node) nogil
    cdef void xmlDocDumpMemory(xmlDoc* cur, char** mem, int* size) nogil
    cdef void xmlDocDumpMemoryEnc(xmlDoc* cur, char** mem, int* size,
                                  char* encoding) nogil
    cdef int xmlSaveFileTo(xmlOutputBuffer* out, xmlDoc* cur,
                           char* encoding) nogil

    cdef void xmlUnlinkNode(xmlNode* cur) nogil
    cdef xmlNode* xmlDocSetRootElement(xmlDoc* doc, xmlNode* root) nogil
    cdef xmlNode* xmlDocGetRootElement(xmlDoc* doc) nogil
    cdef void xmlSetTreeDoc(xmlNode* tree, xmlDoc* doc) nogil
    cdef xmlAttr* xmlHasProp(xmlNode* node, const_xmlChar* name) nogil
    cdef xmlAttr* xmlHasNsProp(xmlNode* node, const_xmlChar* name, const_xmlChar* nameSpace) nogil
    cdef xmlChar* xmlNodeGetContent(xmlNode* cur) nogil
    cdef int xmlNodeBufGetContent(xmlBuffer* buffer, xmlNode* cur) nogil
    cdef xmlNs* xmlSearchNs(xmlDoc* doc, xmlNode* node, const_xmlChar* prefix) nogil
    cdef xmlNs* xmlSearchNsByHref(xmlDoc* doc, xmlNode* node, const_xmlChar* href) nogil
    cdef int xmlIsBlankNode(xmlNode* node) nogil
    cdef long xmlGetLineNo(xmlNode* node) nogil
    cdef void xmlElemDump(stdio.FILE* f, xmlDoc* doc, xmlNode* cur) nogil
    cdef void xmlNodeDumpOutput(xmlOutputBuffer* buf,
                                xmlDoc* doc, xmlNode* cur, int level,
                                int format, const_char* encoding) nogil
    cdef void xmlBufAttrSerializeTxtContent(xmlOutputBuffer *buf, xmlDoc *doc,
                                xmlAttr *attr, const_xmlChar *string) nogil
    cdef void xmlNodeSetName(xmlNode* cur, const_xmlChar* name) nogil
    cdef void xmlNodeSetContent(xmlNode* cur, const_xmlChar* content) nogil
    cdef xmlDtd* xmlCopyDtd(xmlDtd* dtd) nogil
    cdef xmlDoc* xmlCopyDoc(xmlDoc* doc, int recursive) nogil
    cdef xmlNode* xmlCopyNode(xmlNode* node, int extended) nogil
    cdef xmlNode* xmlDocCopyNode(xmlNode* node, xmlDoc* doc, int extended) nogil
    cdef int xmlReconciliateNs(xmlDoc* doc, xmlNode* tree) nogil
    cdef xmlNs* xmlNewReconciliedNs(xmlDoc* doc, xmlNode* tree, xmlNs* ns) nogil
    cdef xmlBuffer* xmlBufferCreate() nogil
    cdef void xmlBufferWriteChar(xmlBuffer* buf, char* string) nogil
    cdef void xmlBufferFree(xmlBuffer* buf) nogil
    cdef const_xmlChar* xmlBufferContent(xmlBuffer* buf) nogil
    cdef int xmlBufferLength(xmlBuffer* buf) nogil
    cdef const_xmlChar* xmlBufContent(xmlBuf* buf) nogil # new in libxml2 2.9
    cdef size_t xmlBufUse(xmlBuf* buf) nogil # new in libxml2 2.9
    cdef int xmlKeepBlanksDefault(int val) nogil
    cdef xmlChar* xmlNodeGetBase(xmlDoc* doc, xmlNode* node) nogil
    cdef xmlDtd* xmlCreateIntSubset(xmlDoc* doc, const_xmlChar* name,
                                    const_xmlChar* ExternalID, const_xmlChar* SystemID) nogil
    cdef void xmlNodeSetBase(xmlNode* node, const_xmlChar* uri) nogil
    cdef int xmlValidateNCName(const_xmlChar* value, int space) nogil

cdef extern from "libxml/uri.h":
    cdef const_xmlChar* xmlBuildURI(const_xmlChar* href, const_xmlChar* base) nogil

cdef extern from "libxml/HTMLtree.h":
    cdef void htmlNodeDumpFormatOutput(xmlOutputBuffer* buf,
                                       xmlDoc* doc, xmlNode* cur,
                                       char* encoding, int format) nogil
    cdef xmlDoc* htmlNewDoc(const_xmlChar* uri, const_xmlChar* externalID) nogil

cdef extern from "libxml/valid.h":
    cdef xmlAttr* xmlGetID(xmlDoc* doc, const_xmlChar* ID) nogil
    cdef void xmlDumpNotationTable(xmlBuffer* buffer,
                                   xmlNotationTable* table) nogil
    cdef int xmlValidateNameValue(const_xmlChar* value) nogil

cdef extern from "libxml/xmlIO.h":
    cdef int xmlOutputBufferWrite(xmlOutputBuffer* out,
                                  int len, const_char* str) nogil
    cdef int xmlOutputBufferWriteString(xmlOutputBuffer* out, const_char* str) nogil
    cdef int xmlOutputBufferWriteEscape(xmlOutputBuffer* out,
                                        const_xmlChar* str,
                                        xmlCharEncodingOutputFunc escapefunc) nogil
    cdef int xmlOutputBufferFlush(xmlOutputBuffer* out) nogil
    cdef int xmlOutputBufferClose(xmlOutputBuffer* out) nogil

    ctypedef int (*xmlInputReadCallback)(void* context,
                                         char* buffer, int len)
    ctypedef int (*xmlInputCloseCallback)(void* context)

    ctypedef int (*xmlOutputWriteCallback)(void* context,
                                           char* buffer, int len)
    ctypedef int (*xmlOutputCloseCallback)(void* context)

    cdef xmlOutputBuffer* xmlAllocOutputBuffer(
        xmlCharEncodingHandler* encoder) nogil
    cdef xmlOutputBuffer* xmlOutputBufferCreateIO(
        xmlOutputWriteCallback iowrite,
        xmlOutputCloseCallback ioclose,
        void * ioctx, 
        xmlCharEncodingHandler* encoder) nogil
    cdef xmlOutputBuffer* xmlOutputBufferCreateFile(
        stdio.FILE* file, xmlCharEncodingHandler* encoder) nogil
    cdef xmlOutputBuffer* xmlOutputBufferCreateFilename(
        char* URI, xmlCharEncodingHandler* encoder, int compression) nogil

cdef extern from "libxml/xmlsave.h":
    ctypedef struct xmlSaveCtxt

    ctypedef enum xmlSaveOption:
        XML_SAVE_FORMAT   = 1   # format save output            (2.6.17)
        XML_SAVE_NO_DECL  = 2   # drop the xml declaration      (2.6.21)
        XML_SAVE_NO_EMPTY = 4   # no empty tags                 (2.6.22)
        XML_SAVE_NO_XHTML = 8   # disable XHTML1 specific rules (2.6.22)
        XML_SAVE_XHTML = 16     # force XHTML1 specific rules         (2.7.2)
        XML_SAVE_AS_XML = 32    # force XML serialization on HTML doc (2.7.2)
        XML_SAVE_AS_HTML = 64   # force HTML serialization on XML doc (2.7.2)

    cdef xmlSaveCtxt* xmlSaveToFilename(char* filename, char* encoding,
                                        int options) nogil
    cdef xmlSaveCtxt* xmlSaveToBuffer(xmlBuffer* buffer, char* encoding,
                                      int options) nogil # libxml2 2.6.23
    cdef long xmlSaveDoc(xmlSaveCtxt* ctxt, xmlDoc* doc) nogil
    cdef long xmlSaveTree(xmlSaveCtxt* ctxt, xmlNode* node) nogil
    cdef int xmlSaveClose(xmlSaveCtxt* ctxt) nogil
    cdef int xmlSaveFlush(xmlSaveCtxt* ctxt) nogil
    cdef int xmlSaveSetAttrEscape(xmlSaveCtxt* ctxt, void* escape_func) nogil
    cdef int xmlSaveSetEscape(xmlSaveCtxt* ctxt, void* escape_func) nogil

cdef extern from "libxml/globals.h":
    cdef int xmlThrDefKeepBlanksDefaultValue(int onoff) nogil
    cdef int xmlThrDefLineNumbersDefaultValue(int onoff) nogil
    cdef int xmlThrDefIndentTreeOutput(int onoff) nogil
    
cdef extern from "libxml/xmlmemory.h" nogil:
    cdef void* xmlMalloc(size_t size)
    cdef int xmlMemBlocks()
    cdef int xmlMemUsed()
    cdef void xmlMemDisplay(stdio.FILE* file)
    cdef void xmlMemDisplayLast(stdio.FILE* file, long num_bytes)
    cdef void xmlMemShow(stdio.FILE* file, int count)

cdef extern from "etree_defs.h":
    cdef bint _isElement(xmlNode* node) nogil
    cdef bint _isElementOrXInclude(xmlNode* node) nogil
    cdef const_xmlChar* _getNs(xmlNode* node) nogil
    cdef void BEGIN_FOR_EACH_ELEMENT_FROM(xmlNode* tree_top,
                                          xmlNode* start_node,
                                          bint inclusive) nogil
    cdef void END_FOR_EACH_ELEMENT_FROM(xmlNode* start_node) nogil
    cdef void BEGIN_FOR_EACH_FROM(xmlNode* tree_top,
                                  xmlNode* start_node,
                                  bint inclusive) nogil
    cdef void END_FOR_EACH_FROM(xmlNode* start_node) nogil
