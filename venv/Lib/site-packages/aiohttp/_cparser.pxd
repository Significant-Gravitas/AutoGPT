from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)


cdef extern from "../vendor/llhttp/build/llhttp.h":

    struct llhttp__internal_s:
        int32_t _index
        void* _span_pos0
        void* _span_cb0
        int32_t error
        const char* reason
        const char* error_pos
        void* data
        void* _current
        uint64_t content_length
        uint8_t type
        uint8_t method
        uint8_t http_major
        uint8_t http_minor
        uint8_t header_state
        uint8_t lenient_flags
        uint8_t upgrade
        uint8_t finish
        uint16_t flags
        uint16_t status_code
        void* settings

    ctypedef llhttp__internal_s llhttp__internal_t
    ctypedef llhttp__internal_t llhttp_t

    ctypedef int (*llhttp_data_cb)(llhttp_t*, const char *at, size_t length) except -1
    ctypedef int (*llhttp_cb)(llhttp_t*) except -1

    struct llhttp_settings_s:
        llhttp_cb      on_message_begin
        llhttp_data_cb on_url
        llhttp_data_cb on_status
        llhttp_data_cb on_header_field
        llhttp_data_cb on_header_value
        llhttp_cb      on_headers_complete
        llhttp_data_cb on_body
        llhttp_cb      on_message_complete
        llhttp_cb      on_chunk_header
        llhttp_cb      on_chunk_complete

        llhttp_cb      on_url_complete
        llhttp_cb      on_status_complete
        llhttp_cb      on_header_field_complete
        llhttp_cb      on_header_value_complete

    ctypedef llhttp_settings_s llhttp_settings_t

    enum llhttp_errno:
        HPE_OK,
        HPE_INTERNAL,
        HPE_STRICT,
        HPE_LF_EXPECTED,
        HPE_UNEXPECTED_CONTENT_LENGTH,
        HPE_CLOSED_CONNECTION,
        HPE_INVALID_METHOD,
        HPE_INVALID_URL,
        HPE_INVALID_CONSTANT,
        HPE_INVALID_VERSION,
        HPE_INVALID_HEADER_TOKEN,
        HPE_INVALID_CONTENT_LENGTH,
        HPE_INVALID_CHUNK_SIZE,
        HPE_INVALID_STATUS,
        HPE_INVALID_EOF_STATE,
        HPE_INVALID_TRANSFER_ENCODING,
        HPE_CB_MESSAGE_BEGIN,
        HPE_CB_HEADERS_COMPLETE,
        HPE_CB_MESSAGE_COMPLETE,
        HPE_CB_CHUNK_HEADER,
        HPE_CB_CHUNK_COMPLETE,
        HPE_PAUSED,
        HPE_PAUSED_UPGRADE,
        HPE_USER

    ctypedef llhttp_errno llhttp_errno_t

    enum llhttp_flags:
        F_CONNECTION_KEEP_ALIVE,
        F_CONNECTION_CLOSE,
        F_CONNECTION_UPGRADE,
        F_CHUNKED,
        F_UPGRADE,
        F_CONTENT_LENGTH,
        F_SKIPBODY,
        F_TRAILING,
        F_TRANSFER_ENCODING

    enum llhttp_lenient_flags:
        LENIENT_HEADERS,
        LENIENT_CHUNKED_LENGTH

    enum llhttp_type:
        HTTP_REQUEST,
        HTTP_RESPONSE,
        HTTP_BOTH

    enum llhttp_finish_t:
        HTTP_FINISH_SAFE,
        HTTP_FINISH_SAFE_WITH_CB,
        HTTP_FINISH_UNSAFE

    enum llhttp_method:
        HTTP_DELETE,
        HTTP_GET,
        HTTP_HEAD,
        HTTP_POST,
        HTTP_PUT,
        HTTP_CONNECT,
        HTTP_OPTIONS,
        HTTP_TRACE,
        HTTP_COPY,
        HTTP_LOCK,
        HTTP_MKCOL,
        HTTP_MOVE,
        HTTP_PROPFIND,
        HTTP_PROPPATCH,
        HTTP_SEARCH,
        HTTP_UNLOCK,
        HTTP_BIND,
        HTTP_REBIND,
        HTTP_UNBIND,
        HTTP_ACL,
        HTTP_REPORT,
        HTTP_MKACTIVITY,
        HTTP_CHECKOUT,
        HTTP_MERGE,
        HTTP_MSEARCH,
        HTTP_NOTIFY,
        HTTP_SUBSCRIBE,
        HTTP_UNSUBSCRIBE,
        HTTP_PATCH,
        HTTP_PURGE,
        HTTP_MKCALENDAR,
        HTTP_LINK,
        HTTP_UNLINK,
        HTTP_SOURCE,
        HTTP_PRI,
        HTTP_DESCRIBE,
        HTTP_ANNOUNCE,
        HTTP_SETUP,
        HTTP_PLAY,
        HTTP_PAUSE,
        HTTP_TEARDOWN,
        HTTP_GET_PARAMETER,
        HTTP_SET_PARAMETER,
        HTTP_REDIRECT,
        HTTP_RECORD,
        HTTP_FLUSH

    ctypedef llhttp_method llhttp_method_t;

    void llhttp_settings_init(llhttp_settings_t* settings)
    void llhttp_init(llhttp_t* parser, llhttp_type type,
                 const llhttp_settings_t* settings)

    llhttp_errno_t llhttp_execute(llhttp_t* parser, const char* data, size_t len)
    llhttp_errno_t llhttp_finish(llhttp_t* parser)

    int llhttp_message_needs_eof(const llhttp_t* parser)

    int llhttp_should_keep_alive(const llhttp_t* parser)

    void llhttp_pause(llhttp_t* parser)
    void llhttp_resume(llhttp_t* parser)

    void llhttp_resume_after_upgrade(llhttp_t* parser)

    llhttp_errno_t llhttp_get_errno(const llhttp_t* parser)
    const char* llhttp_get_error_reason(const llhttp_t* parser)
    void llhttp_set_error_reason(llhttp_t* parser, const char* reason)
    const char* llhttp_get_error_pos(const llhttp_t* parser)
    const char* llhttp_errno_name(llhttp_errno_t err)

    const char* llhttp_method_name(llhttp_method_t method)

    void llhttp_set_lenient_headers(llhttp_t* parser, int enabled)
    void llhttp_set_lenient_chunked_length(llhttp_t* parser, int enabled)
