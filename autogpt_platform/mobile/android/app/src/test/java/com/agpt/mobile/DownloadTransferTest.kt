package com.agpt.mobile

import java.io.ByteArrayOutputStream
import java.util.Base64
import org.junit.Assert.*
import org.junit.Test

class DownloadTransferTest {
    @Test
    fun permitsOnlyTrustedMainFrameMessages() {
        val origin = ServerOrigin.parse("https://platform.agpt.co", false)!!
        assertTrue(
            DownloadMessage.trusted(
                origin,
                "https://platform.agpt.co",
                "https://platform.agpt.co/copilot",
                true,
            )
        )
        assertFalse(
            DownloadMessage.trusted(
                origin,
                "https://platform.agpt.co",
                "https://platform.agpt.co/copilot",
                false,
            )
        )
        assertFalse(
            DownloadMessage.trusted(
                origin,
                "https://platform.agpt.co.attacker.test",
                "https://platform.agpt.co/copilot",
                true,
            )
        )
        assertFalse(
            DownloadMessage.trusted(
                origin,
                "https://platform.agpt.co",
                "https://attacker.test",
                true,
            )
        )
    }

    @Test
    fun rejectsMalformedOversizedAndInvalidMessages() {
        listOf(
                "not-json",
                "[]",
                "{}",
                start() + "garbage",
                "{\"type\":\"start\",\"id\":\"a\"}",
                start(size = 50 * 1024 * 1024 + 1),
                start(size = -1),
                start(filename = "../private.txt"),
                start(filename = ""),
                start(mimeType = "anything"),
                start().replace("\"size\":3", "\"size\":3.5"),
                "{\"type\":\"chunk\",\"id\":\"a\",\"index\":0,\"data\":\"@@@\"}",
                "{\"type\":\"chunk\",\"id\":\"a\",\"index\":-1,\"data\":\"YWJj\"}",
                " ".repeat(DownloadMessage.MAX_MESSAGE_BYTES + 1),
            )
            .forEach { raw ->
                assertThrows(IllegalArgumentException::class.java) { DownloadMessage.parse(raw) }
            }
        val oversized =
            Base64.getEncoder().encodeToString(ByteArray(DownloadMessage.MAX_CHUNK_BYTES + 1))
        assertThrows(IllegalArgumentException::class.java) {
            DownloadMessage.parse(
                "{\"type\":\"chunk\",\"id\":\"a\",\"index\":0,\"data\":\"$oversized\"}"
            )
        }
    }

    @Test
    fun writesOnlySequentialBytesAndCompletesExactSize() {
        val output = ByteArrayOutputStream()
        val transfer =
            DownloadTransfer(DownloadMessage.parse(start()) as DownloadMessage.Start, output)
        transfer.append(0, byteArrayOf(1, 2))
        transfer.append(1, byteArrayOf(3))
        transfer.finish()
        assertArrayEquals(byteArrayOf(1, 2, 3), output.toByteArray())
        assertThrows(IllegalArgumentException::class.java) { transfer.append(2, byteArrayOf(4)) }
    }

    @Test
    fun rejectsOutOfOrderOversizedPrematureAndCanceledWrites() {
        fun transfer() =
            DownloadTransfer(
                DownloadMessage.parse(start()) as DownloadMessage.Start,
                ByteArrayOutputStream(),
            )
        assertThrows(IllegalArgumentException::class.java) { transfer().append(1, byteArrayOf(1)) }
        assertThrows(IllegalArgumentException::class.java) {
            transfer().append(0, byteArrayOf(1, 2, 3, 4))
        }
        assertThrows(IllegalArgumentException::class.java) { transfer().append(0, byteArrayOf()) }
        assertThrows(IllegalArgumentException::class.java) { transfer().finish() }
        val canceled = transfer()
        canceled.abort()
        assertThrows(IllegalArgumentException::class.java) { canceled.append(0, byteArrayOf(1)) }
        assertThrows(IllegalArgumentException::class.java) { canceled.finish() }
    }

    @Test
    fun invalidMetadataStillHasACorrelatableSafeRequestId() {
        assertEquals(
            "request-123",
            DownloadMessage.requestId("{\"id\":\"request-123\",\"size\":-1}"),
        )
        assertNull(DownloadMessage.requestId("{\"id\":\"../file\"}"))
        assertNull(DownloadMessage.requestId("not-json"))
    }

    @Test
    fun permitsTheSameMimeTokenCharactersAsTheWebHelper() {
        assertTrue(
            DownloadMessage.parse(start(mimeType = "application/vnd.example!#$&^_+.-"))
                is DownloadMessage.Start
        )
    }

    @Test
    fun zeroLengthFilesCanFinishWithoutAChunk() {
        val transfer =
            DownloadTransfer(
                DownloadMessage.parse(start(size = 0)) as DownloadMessage.Start,
                ByteArrayOutputStream(),
            )
        transfer.finish()
    }

    private fun start(
        filename: String = "result.txt",
        mimeType: String = "text/plain",
        size: Int = 3,
    ) =
        "{\"type\":\"start\",\"id\":\"a\",\"filename\":\"$filename\",\"mimeType\":\"$mimeType\",\"size\":$size}"
}
