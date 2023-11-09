from lxml.includes.tree cimport xmlDoc, xmlNode, xmlDict, xmlChar, const_xmlChar, xmlOutputBuffer
from lxml.includes.xmlerror cimport xmlGenericErrorFunc
from lxml.includes.xpath cimport xmlXPathContext, xmlXPathFunction

from libc.string cimport const_char

cdef extern from "libxslt/xslt.h":
    cdef int xsltLibxsltVersion
    cdef int xsltMaxDepth

cdef extern from "libxslt/xsltconfig.h":
    cdef int LIBXSLT_VERSION

cdef extern from "libxslt/xsltInternals.h":
    ctypedef enum xsltTransformState:
        XSLT_STATE_OK       # 0
        XSLT_STATE_ERROR    # 1
        XSLT_STATE_STOPPED  # 2

    ctypedef struct xsltDocument:
        xmlDoc* doc

    ctypedef struct xsltStylesheet:
        xmlChar* encoding
        xmlDoc* doc
        int errors

    ctypedef struct xsltTransformContext:
        xsltStylesheet* style
        xmlXPathContext* xpathCtxt
        xsltDocument* document
        void* _private
        xmlDict* dict
        int profile
        xmlNode* node
        xmlDoc* output
        xmlNode* insert
        xmlNode* inst
        xsltTransformState state

    ctypedef struct xsltStackElem

    ctypedef struct xsltTemplate

    cdef xsltStylesheet* xsltParseStylesheetDoc(xmlDoc* doc) nogil
    cdef void xsltFreeStylesheet(xsltStylesheet* sheet) nogil

cdef extern from "libxslt/imports.h":
    # actually defined in "etree_defs.h"
    cdef void LXML_GET_XSLT_ENCODING(const_xmlChar* result_var, xsltStylesheet* style)

cdef extern from "libxslt/extensions.h":
    ctypedef void (*xsltTransformFunction)(xsltTransformContext* ctxt,
                                           xmlNode* context_node,
                                           xmlNode* inst,
                                           void* precomp_unused) nogil

    cdef int xsltRegisterExtFunction(xsltTransformContext* ctxt,
                                     const_xmlChar* name,
                                     const_xmlChar* URI,
                                     xmlXPathFunction function) nogil
    cdef int xsltRegisterExtModuleFunction(const_xmlChar* name, const_xmlChar* URI,
                                           xmlXPathFunction function) nogil
    cdef int xsltUnregisterExtModuleFunction(const_xmlChar* name, const_xmlChar* URI)
    cdef xmlXPathFunction xsltExtModuleFunctionLookup(
        const_xmlChar* name, const_xmlChar* URI) nogil
    cdef int xsltRegisterExtPrefix(xsltStylesheet* style, 
                                   const_xmlChar* prefix, const_xmlChar* URI) nogil
    cdef int xsltRegisterExtElement(xsltTransformContext* ctxt,
                                    const_xmlChar* name, const_xmlChar* URI,
                                    xsltTransformFunction function) nogil

cdef extern from "libxslt/documents.h":
    ctypedef enum xsltLoadType:
        XSLT_LOAD_START
        XSLT_LOAD_STYLESHEET
        XSLT_LOAD_DOCUMENT

    ctypedef xmlDoc* (*xsltDocLoaderFunc)(const_xmlChar* URI, xmlDict* dict,
                                          int options,
                                          void* ctxt,
                                          xsltLoadType type) nogil
    cdef xsltDocLoaderFunc xsltDocDefaultLoader
    cdef void xsltSetLoaderFunc(xsltDocLoaderFunc f) nogil

cdef extern from "libxslt/transform.h":
    cdef xmlDoc* xsltApplyStylesheet(xsltStylesheet* style, xmlDoc* doc,
                                     const_char** params) nogil
    cdef xmlDoc* xsltApplyStylesheetUser(xsltStylesheet* style, xmlDoc* doc,
                                         const_char** params, const_char* output,
                                         void* profile,
                                         xsltTransformContext* context) nogil
    cdef void xsltProcessOneNode(xsltTransformContext* ctxt,
                                 xmlNode* contextNode,
                                 xsltStackElem* params) nogil
    cdef xsltTransformContext* xsltNewTransformContext(xsltStylesheet* style,
                                                       xmlDoc* doc) nogil
    cdef void xsltFreeTransformContext(xsltTransformContext* context) nogil
    cdef void xsltApplyOneTemplate(xsltTransformContext* ctxt,
                                   xmlNode* contextNode, xmlNode* list,
                                   xsltTemplate* templ,
                                   xsltStackElem* params) nogil


cdef extern from "libxslt/xsltutils.h":
    cdef int xsltSaveResultToString(xmlChar** doc_txt_ptr,
                                    int* doc_txt_len,
                                    xmlDoc* result,
                                    xsltStylesheet* style) nogil
    cdef int xsltSaveResultToFilename(const_char *URL,
                                      xmlDoc* result,
                                      xsltStylesheet* style,
                                      int compression) nogil
    cdef int xsltSaveResultTo(xmlOutputBuffer* buf,
                              xmlDoc* result,
                              xsltStylesheet* style) nogil
    cdef xmlGenericErrorFunc xsltGenericError
    cdef void *xsltGenericErrorContext
    cdef void xsltSetGenericErrorFunc(
        void* ctxt, void (*handler)(void* ctxt, char* msg, ...)) nogil
    cdef void xsltSetTransformErrorFunc(
        xsltTransformContext*, void* ctxt,
        void (*handler)(void* ctxt, char* msg, ...) nogil) nogil
    cdef void xsltTransformError(xsltTransformContext* ctxt, 
                                 xsltStylesheet* style, 
                                 xmlNode* node, char* msg, ...)
    cdef void xsltSetCtxtParseOptions(
        xsltTransformContext* ctxt, int options)


cdef extern from "libxslt/security.h":
    ctypedef struct xsltSecurityPrefs
    ctypedef enum xsltSecurityOption:
        XSLT_SECPREF_READ_FILE = 1
        XSLT_SECPREF_WRITE_FILE = 2
        XSLT_SECPREF_CREATE_DIRECTORY = 3
        XSLT_SECPREF_READ_NETWORK = 4
        XSLT_SECPREF_WRITE_NETWORK = 5

    ctypedef int (*xsltSecurityCheck)(xsltSecurityPrefs* sec,
                                      xsltTransformContext* ctxt,
                                      char* value) nogil

    cdef xsltSecurityPrefs* xsltNewSecurityPrefs() nogil
    cdef void xsltFreeSecurityPrefs(xsltSecurityPrefs* sec) nogil
    cdef int xsltSecurityForbid(xsltSecurityPrefs* sec,
                                xsltTransformContext* ctxt,
                                char* value) nogil
    cdef int xsltSecurityAllow(xsltSecurityPrefs* sec,
                                xsltTransformContext* ctxt,
                                char* value) nogil
    cdef int xsltSetSecurityPrefs(xsltSecurityPrefs* sec,
                                  xsltSecurityOption option,
                                  xsltSecurityCheck func) nogil
    cdef xsltSecurityCheck xsltGetSecurityPrefs(
        xsltSecurityPrefs* sec,
        xsltSecurityOption option) nogil
    cdef int xsltSetCtxtSecurityPrefs(xsltSecurityPrefs* sec,
                                      xsltTransformContext* ctxt) nogil
    cdef xmlDoc* xsltGetProfileInformation(xsltTransformContext* ctxt) nogil

cdef extern from "libxslt/variables.h":
    cdef int xsltQuoteUserParams(xsltTransformContext* ctxt,
                                 const_char** params)
    cdef int xsltQuoteOneUserParam(xsltTransformContext* ctxt,
                                   const_xmlChar* name,
                                   const_xmlChar* value)

cdef extern from "libxslt/extra.h":
    const_xmlChar* XSLT_LIBXSLT_NAMESPACE
    const_xmlChar* XSLT_XALAN_NAMESPACE
    const_xmlChar* XSLT_SAXON_NAMESPACE
    const_xmlChar* XSLT_XT_NAMESPACE

    cdef xmlXPathFunction xsltFunctionNodeSet
    cdef void xsltRegisterAllExtras() nogil

cdef extern from "libexslt/exslt.h":
    cdef void exsltRegisterAll() nogil

    # libexslt 1.1.25+
    const_xmlChar* EXSLT_DATE_NAMESPACE
    const_xmlChar* EXSLT_SETS_NAMESPACE
    const_xmlChar* EXSLT_MATH_NAMESPACE
    const_xmlChar* EXSLT_STRINGS_NAMESPACE

    cdef int exsltDateXpathCtxtRegister(xmlXPathContext* ctxt, const_xmlChar* prefix)
    cdef int exsltSetsXpathCtxtRegister(xmlXPathContext* ctxt, const_xmlChar* prefix)
    cdef int exsltMathXpathCtxtRegister(xmlXPathContext* ctxt, const_xmlChar* prefix)
    cdef int exsltStrXpathCtxtRegister(xmlXPathContext* ctxt, const_xmlChar* prefix)

