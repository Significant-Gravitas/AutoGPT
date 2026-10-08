package com.agpt.mobile

import org.junit.Assert.*
import org.junit.Test

class PushTargetTest {
    private val origin = ServerOrigin.parse("https://platform.agpt.co", false)!!

    @Test
    fun routesAreBoundToTheCurrentAccountAndOrigin() {
        assertEquals(
            "https://platform.agpt.co/home?sessionId=hello",
            PushTarget.url("/home?sessionId=hello", origin.value, "current", "current", origin),
        )
        assertNull(PushTarget.url("/mobile?tab=attention", origin.value, "old", "current", origin))
        assertNull(
            PushTarget.url(
                "/mobile?tab=attention",
                "https://other.example",
                "current",
                "current",
                origin,
            )
        )
        listOf(
                "https://attacker.example",
                "//attacker.example/home",
                "/api/auth/sign-out",
                "/home?next=evil",
                "/home?sessionId=one&sessionId=two",
            )
            .forEach {
                assertNull(PushTarget.url(it, origin.value, "current", "current", origin))
            }
    }
}
