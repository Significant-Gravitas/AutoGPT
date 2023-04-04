
@cython.final
@cython.internal
cdef class _MemDebug:
    """Debugging support for the memory allocation in libxml2.
    """
    def bytes_used(self):
        """bytes_used(self)

        Returns the total amount of memory (in bytes) currently used by libxml2.
        Note that libxml2 constrains this value to a C int, which limits
        the accuracy on 64 bit systems.
        """
        return tree.xmlMemUsed()

    def blocks_used(self):
        """blocks_used(self)

        Returns the total number of memory blocks currently allocated by libxml2.
        Note that libxml2 constrains this value to a C int, which limits
        the accuracy on 64 bit systems.
        """
        return tree.xmlMemBlocks()

    def dict_size(self):
        """dict_size(self)

        Returns the current size of the global name dictionary used by libxml2
        for the current thread.  Each thread has its own dictionary.
        """
        c_dict = __GLOBAL_PARSER_CONTEXT._getThreadDict(NULL)
        if c_dict is NULL:
            raise MemoryError()
        return tree.xmlDictSize(c_dict)

    def dump(self, output_file=None, byte_count=None):
        """dump(self, output_file=None, byte_count=None)

        Dumps the current memory blocks allocated by libxml2 to a file.

        The optional parameter 'output_file' specifies the file path.  It defaults
        to the file ".memorylist" in the current directory.

        The optional parameter 'byte_count' limits the number of bytes in the dump.
        Note that this parameter is ignored when lxml is compiled against a libxml2
        version before 2.7.0.
        """
        cdef Py_ssize_t c_count
        if output_file is None:
            output_file = b'.memorylist'
        elif isinstance(output_file, unicode):
            output_file.encode(sys.getfilesystemencoding())

        f = stdio.fopen(output_file, "w")
        if f is NULL:
            raise IOError(f"Failed to create file {output_file.decode(sys.getfilesystemencoding())}")
        try:
            if byte_count is None:
                tree.xmlMemDisplay(f)
            else:
                c_count = byte_count
                tree.xmlMemDisplayLast(f, c_count)
        finally:
            stdio.fclose(f)

    def show(self, output_file=None, block_count=None):
        """show(self, output_file=None, block_count=None)

        Dumps the current memory blocks allocated by libxml2 to a file.
        The output file format is suitable for line diffing.

        The optional parameter 'output_file' specifies the file path.  It defaults
        to the file ".memorydump" in the current directory.

        The optional parameter 'block_count' limits the number of blocks
        in the dump.
        """
        if output_file is None:
            output_file = b'.memorydump'
        elif isinstance(output_file, unicode):
            output_file.encode(sys.getfilesystemencoding())

        f = stdio.fopen(output_file, "w")
        if f is NULL:
            raise IOError(f"Failed to create file {output_file.decode(sys.getfilesystemencoding())}")
        try:
            tree.xmlMemShow(f, block_count if block_count is not None else tree.xmlMemBlocks())
        finally:
            stdio.fclose(f)

memory_debugger = _MemDebug()
