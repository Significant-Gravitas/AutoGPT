cdef extern from "libxml/uri.h":
    ctypedef struct xmlURI

    cdef xmlURI* xmlParseURI(char* str)
    cdef void xmlFreeURI(xmlURI* uri)
