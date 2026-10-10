package com.agpt.mobile

import org.junit.Assert.*
import org.junit.Test

class ServerOriginTest {
    @Test
    fun normalizesAnExplicitHttpsOrigin() {
        val origin = ServerOrigin.parse(" https://PLATFORM.agpt.co:443/ ", false)!!
        assertEquals("https://platform.agpt.co", origin.value)
        assertTrue(origin.contains("https://platform.agpt.co/copilot?session=123"))
        assertEquals("https://platform.agpt.co/copilot", origin.chatUrl)
    }

    @Test
    fun rejectsEveryOtherOriginAndDangerousUri() {
        val origin = ServerOrigin.parse("https://platform.agpt.co", false)!!
        listOf(
                "https://platform.agpt.co.attacker.test/copilot",
                "https://attacker.test@platform.agpt.co/copilot",
                "https://platform.agpt.co:444/copilot",
                "http://platform.agpt.co/copilot",
                "javascript:alert(1)",
                "file:///etc/passwd",
                "https://platform.agpt.co./copilot",
                "https://platform.agpt.co\\@attacker.test",
                "//platform.agpt.co/copilot",
            )
            .forEach { assertFalse(it, origin.contains(it)) }
    }

    @Test
    fun customServersMustBeDeliberateBareOrigins() {
        assertNotNull(ServerOrigin.parse("https://preview.example.test:8443", false))
        listOf(
                "https://example.test/path",
                "https://example.test?x=1",
                "https://example.test#x",
                "https://user:password@example.test", // pragma: allowlist secret
                "example.test",
                "https://example.test:0",
                "https://example.test:65536",
                "https://example.test:",
                "https://example.test.",
            )
            .forEach { assertNull(it, ServerOrigin.parse(it, false)) }
    }

    @Test
    fun embedsPreserveWebFeaturesWithoutExpandingTheMainFrameBoundary() {
        val origin = ServerOrigin.parse("https://platform.agpt.co", false)!!
        listOf(
                "https://www.youtube.com/embed/video",
                "https://player.vimeo.com/video/1",
                "blob:https://platform.agpt.co/uuid",
                "about:blank",
                "about:srcdoc",
            )
            .forEach {
                assertTrue(it, origin.allowsEmbeddedUrl(it))
                assertFalse(it, origin.contains(it))
            }
        listOf(
                "blob:https://attacker.test/uuid",
                "http://example.test",
                "file:///secret",
                "content://files/private",
                "javascript:alert(1)",
            )
            .forEach { assertFalse(it, origin.allowsEmbeddedUrl(it)) }
    }

    @Test
    fun cleartextIsLimitedToDebugLoopbackHosts() {
        listOf("localhost", "127.0.0.1", "10.0.2.2").forEach {
            assertNotNull(ServerOrigin.parse("http://$it:8765", true))
            assertNull(ServerOrigin.parse("http://$it:8765", false))
        }
        assertNull(ServerOrigin.parse("http://192.168.1.2:8765", true))
        assertNull(ServerOrigin.parse("http://localhost.attacker.test", true))
    }
}
