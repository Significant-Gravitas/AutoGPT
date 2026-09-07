package com.agpt.mobile

import java.io.OutputStream

class DownloadTransfer(
    private val request: DownloadMessage.Start,
    private val output: OutputStream,
) {
    private var index = 0
    private var written = 0L
    private var closed = false

    fun append(chunkIndex: Int, bytes: ByteArray) {
        require(!closed && chunkIndex == index)
        require(bytes.isNotEmpty() && bytes.size <= DownloadMessage.MAX_CHUNK_BYTES)
        require(written + bytes.size <= request.size)
        output.write(bytes)
        written += bytes.size
        index++
    }

    fun finish() {
        require(!closed && written == request.size)
        output.flush()
        output.close()
        closed = true
    }

    fun abort() {
        if (!closed) {
            closed = true
            runCatching { output.close() }
        }
    }
}
