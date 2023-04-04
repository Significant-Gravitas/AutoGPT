from lxml.includes.tree cimport xmlDoc, xmlNode

cdef extern from "libxml/xinclude.h":

    ctypedef struct xmlXIncludeCtxt

    cdef int xmlXIncludeProcess(xmlDoc* doc) nogil
    cdef int xmlXIncludeProcessFlags(xmlDoc* doc, int parser_opts) nogil
    cdef int xmlXIncludeProcessTree(xmlNode* doc) nogil
    cdef int xmlXIncludeProcessTreeFlags(xmlNode* doc, int parser_opts) nogil

    # libxml2 >= 2.7.4
    cdef int xmlXIncludeProcessTreeFlagsData(
            xmlNode* doc, int parser_opts, void* data) nogil

    cdef xmlXIncludeCtxt* xmlXIncludeNewContext(xmlDoc* doc) nogil
    cdef int xmlXIncludeProcessNode(xmlXIncludeCtxt* ctxt, xmlNode* node) nogil
    cdef int xmlXIncludeSetFlags(xmlXIncludeCtxt* ctxt, int flags) nogil

    # libxml2 >= 2.6.27
    cdef int xmlXIncludeProcessFlagsData(
        xmlDoc* doc, int flags, void* data) nogil
