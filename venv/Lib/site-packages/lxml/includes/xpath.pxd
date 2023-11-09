from lxml.includes cimport tree
from lxml.includes cimport xmlerror

from libc.string cimport const_char
from lxml.includes.tree cimport xmlChar, const_xmlChar

cdef extern from "libxml/xpath.h":
    ctypedef enum xmlXPathObjectType:
        XPATH_UNDEFINED = 0
        XPATH_NODESET = 1
        XPATH_BOOLEAN = 2
        XPATH_NUMBER = 3
        XPATH_STRING = 4
        XPATH_POINT = 5
        XPATH_RANGE = 6
        XPATH_LOCATIONSET = 7
        XPATH_USERS = 8
        XPATH_XSLT_TREE = 9

    ctypedef enum xmlXPathError:
        XPATH_EXPRESSION_OK = 0
        XPATH_NUMBER_ERROR = 1
        XPATH_UNFINISHED_LITERAL_ERROR = 2
        XPATH_START_LITERAL_ERROR = 3
        XPATH_VARIABLE_REF_ERROR = 4
        XPATH_UNDEF_VARIABLE_ERROR = 5
        XPATH_INVALID_PREDICATE_ERROR = 6
        XPATH_EXPR_ERROR = 7
        XPATH_UNCLOSED_ERROR = 8
        XPATH_UNKNOWN_FUNC_ERROR = 9
        XPATH_INVALID_OPERAND = 10
        XPATH_INVALID_TYPE = 11
        XPATH_INVALID_ARITY = 12
        XPATH_INVALID_CTXT_SIZE = 13
        XPATH_INVALID_CTXT_POSITION = 14
        XPATH_MEMORY_ERROR = 15
        XPTR_SYNTAX_ERROR = 16
        XPTR_RESOURCE_ERROR = 17
        XPTR_SUB_RESOURCE_ERROR = 18
        XPATH_UNDEF_PREFIX_ERROR = 19
        XPATH_ENCODING_ERROR = 20
        XPATH_INVALID_CHAR_ERROR = 21
        XPATH_INVALID_CTXT = 22

    ctypedef struct xmlNodeSet:
        int nodeNr
        int nodeMax
        tree.xmlNode** nodeTab
        
    ctypedef struct xmlXPathObject:
        xmlXPathObjectType type
        xmlNodeSet* nodesetval
        bint boolval
        double floatval
        xmlChar* stringval

    ctypedef struct xmlXPathContext:
        tree.xmlDoc* doc
        tree.xmlNode* node
        tree.xmlDict* dict
        tree.xmlHashTable* nsHash
        const_xmlChar* function
        const_xmlChar* functionURI
        xmlerror.xmlStructuredErrorFunc error
        xmlerror.xmlError lastError
        void* userData

    ctypedef struct xmlXPathParserContext:
        xmlXPathContext* context
        xmlXPathObject* value
        tree.xmlNode* ancestor
        int error

    ctypedef struct xmlXPathCompExpr

    ctypedef void (*xmlXPathFunction)(xmlXPathParserContext* ctxt, int nargs) nogil
    ctypedef xmlXPathFunction (*xmlXPathFuncLookupFunc)(void* ctxt,
                                                        const_xmlChar* name,
                                                        const_xmlChar* ns_uri) nogil
    
    cdef xmlXPathContext* xmlXPathNewContext(tree.xmlDoc* doc) nogil
    cdef xmlXPathObject* xmlXPathEvalExpression(const_xmlChar* str,
                                                xmlXPathContext* ctxt) nogil
    cdef xmlXPathObject* xmlXPathCompiledEval(xmlXPathCompExpr* comp,
                                              xmlXPathContext* ctxt) nogil
    cdef xmlXPathCompExpr* xmlXPathCompile(const_xmlChar* str) nogil
    cdef xmlXPathCompExpr* xmlXPathCtxtCompile(xmlXPathContext* ctxt,
                                               const_xmlChar* str) nogil
    cdef void xmlXPathFreeContext(xmlXPathContext* ctxt) nogil
    cdef void xmlXPathFreeCompExpr(xmlXPathCompExpr* comp) nogil
    cdef void xmlXPathFreeObject(xmlXPathObject* obj) nogil
    cdef int xmlXPathRegisterNs(xmlXPathContext* ctxt,
                                const_xmlChar* prefix, const_xmlChar* ns_uri) nogil
    
    cdef xmlNodeSet* xmlXPathNodeSetCreate(tree.xmlNode* val) nogil
    cdef void xmlXPathFreeNodeSet(xmlNodeSet* val) nogil


cdef extern from "libxml/xpathInternals.h":
    cdef int xmlXPathRegisterFunc(xmlXPathContext* ctxt,
                                  const_xmlChar* name,
                                  xmlXPathFunction f) nogil
    cdef int xmlXPathRegisterFuncNS(xmlXPathContext* ctxt,
                                    const_xmlChar* name,
                                    const_xmlChar* ns_uri,
                                    xmlXPathFunction f) nogil
    cdef void xmlXPathRegisterFuncLookup(xmlXPathContext *ctxt,
                                         xmlXPathFuncLookupFunc f,
                                         void *funcCtxt) nogil
    cdef int xmlXPathRegisterVariable(xmlXPathContext *ctxt, 
                                      const_xmlChar* name,
                                      xmlXPathObject* value) nogil
    cdef int xmlXPathRegisterVariableNS(xmlXPathContext *ctxt, 
                                        const_xmlChar* name,
                                        const_xmlChar* ns_uri,
                                        xmlXPathObject* value) nogil
    cdef void xmlXPathRegisteredVariablesCleanup(xmlXPathContext *ctxt) nogil
    cdef void xmlXPathRegisteredNsCleanup(xmlXPathContext *ctxt) nogil
    cdef xmlXPathObject* valuePop (xmlXPathParserContext *ctxt) nogil
    cdef int valuePush(xmlXPathParserContext* ctxt, xmlXPathObject *value) nogil
    
    cdef xmlXPathObject* xmlXPathNewCString(const_char *val) nogil
    cdef xmlXPathObject* xmlXPathWrapCString(const_char * val) nogil
    cdef xmlXPathObject* xmlXPathNewString(const_xmlChar *val) nogil
    cdef xmlXPathObject* xmlXPathWrapString(const_xmlChar * val) nogil
    cdef xmlXPathObject* xmlXPathNewFloat(double val) nogil
    cdef xmlXPathObject* xmlXPathNewBoolean(int val) nogil
    cdef xmlXPathObject* xmlXPathNewNodeSet(tree.xmlNode* val) nogil
    cdef xmlXPathObject* xmlXPathNewValueTree(tree.xmlNode* val) nogil
    cdef void xmlXPathNodeSetAdd(xmlNodeSet* cur,
                                  tree.xmlNode* val) nogil
    cdef void xmlXPathNodeSetAddUnique(xmlNodeSet* cur,
                                        tree.xmlNode* val) nogil
    cdef xmlXPathObject* xmlXPathWrapNodeSet(xmlNodeSet* val) nogil
    cdef void xmlXPathErr(xmlXPathParserContext* ctxt, int error) nogil
