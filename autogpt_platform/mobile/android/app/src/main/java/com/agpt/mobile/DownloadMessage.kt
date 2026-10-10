package com.agpt.mobile

import java.util.Base64
import org.json.JSONObject
import org.json.JSONTokener

sealed interface DownloadMessage {
    val id: String

    data class Start(
        override val id: String,
        val filename: String,
        val mimeType: String,
        val size: Long,
    ) : DownloadMessage

    data class Chunk(override val id: String, val index: Int, val bytes: ByteArray) :
        DownloadMessage

    data class Finish(override val id: String) : DownloadMessage

    data class Cancel(override val id: String) : DownloadMessage

    companion object {
        const val MAX_FILE_BYTES = 50L * 1024 * 1024
        const val MAX_CHUNK_BYTES = 64 * 1024
        const val MAX_MESSAGE_BYTES = 90 * 1024

        fun trusted(
            origin: ServerOrigin,
            source: String,
            currentPage: String?,
            mainFrame: Boolean,
        ): Boolean =
            mainFrame &&
                currentPage != null &&
                origin.contains(source) &&
                origin.contains(currentPage)

        fun parse(raw: String): DownloadMessage =
            try {
                require(
                    raw.length <= MAX_MESSAGE_BYTES &&
                        raw.toByteArray(Charsets.UTF_8).size <= MAX_MESSAGE_BYTES
                )
                val parser = JSONTokener(raw)
                val json = requireNotNull(parser.nextValue() as? JSONObject)
                require(parser.nextClean() == '\u0000')
                val id = string(json, "id")
                require(id.matches(Regex("[A-Za-z0-9_-]{1,128}")))
                when (string(json, "type")) {
                    "start" -> {
                        requireFields(json, setOf("type", "id", "filename", "mimeType", "size"))
                        val filename = string(json, "filename")
                        require(
                            filename.isNotBlank() &&
                                filename.length <= 240 &&
                                filename !in setOf(".", "..")
                        )
                        require(filename.none { it.isISOControl() || it == '/' || it == '\\' })
                        val mimeType = string(json, "mimeType")
                        require(
                            mimeType.length <= 128 &&
                                mimeType.matches(
                                    Regex("[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+")
                                )
                        )
                        val size = integer(json, "size")
                        require(size in 0..MAX_FILE_BYTES)
                        Start(id, filename, mimeType, size)
                    }
                    "chunk" -> {
                        requireFields(json, setOf("type", "id", "index", "data"))
                        val index = integer(json, "index")
                        require(index in 0..Int.MAX_VALUE)
                        val encoded = string(json, "data")
                        require(encoded.length <= ((MAX_CHUNK_BYTES + 2) / 3) * 4)
                        val bytes = Base64.getDecoder().decode(encoded)
                        require(bytes.isNotEmpty() && bytes.size <= MAX_CHUNK_BYTES)
                        require(Base64.getEncoder().encodeToString(bytes) == encoded)
                        Chunk(id, index.toInt(), bytes)
                    }
                    "finish" -> {
                        requireFields(json, setOf("type", "id"))
                        Finish(id)
                    }
                    "cancel" -> {
                        requireFields(json, setOf("type", "id"))
                        Cancel(id)
                    }
                    else -> throw IllegalArgumentException("Unsupported download request")
                }
            } catch (error: Exception) {
                throw IllegalArgumentException("Invalid download request", error)
            }

        fun requestId(raw: String): String? {
            if (raw.length > MAX_MESSAGE_BYTES) return null
            return runCatching { JSONObject(raw).opt("id") as? String }
                .getOrNull()
                ?.takeIf { it.matches(Regex("[A-Za-z0-9_-]{1,128}")) }
        }

        private fun string(json: JSONObject, name: String): String =
            requireNotNull(json.opt(name) as? String)

        private fun integer(json: JSONObject, name: String): Long {
            val value = requireNotNull(json.opt(name) as? Number)
            require(value.toDouble().isFinite() && value.toDouble() == value.toLong().toDouble())
            return value.toLong()
        }

        private fun requireFields(json: JSONObject, expected: Set<String>) {
            require(json.keys().asSequence().toSet() == expected)
        }
    }
}
