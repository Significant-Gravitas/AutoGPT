package com.agpt.mobile

import org.junit.Assert.*
import org.junit.Test

class SessionCookieTest {
    private val origin = ServerOrigin.parse("https://platform.agpt.co", false)!!

    @Test
    fun acceptsHostCookiesAndCanonicalizesAnExactDomain() {
        assertEquals(
            "session=abc; Path=/; Secure; HttpOnly; SameSite=Lax",
            SessionCookie.forOrigin("session=abc; Path=/; Secure; HttpOnly; SameSite=Lax", origin),
        )
        assertEquals(
            "session=abc; Path=/; Secure; HttpOnly",
            SessionCookie.forOrigin(
                "session=abc; Domain=platform.agpt.co; Path=/; Secure; HttpOnly",
                origin,
            ),
        )
    }

    @Test
    fun rejectsCookieDomainEscapesAndMissingProtection() {
        listOf(
                "session=abc; Domain=agpt.co; Path=/; Secure; HttpOnly",
                "session=abc; Domain=attacker.test; Path=/; Secure; HttpOnly",
                "session=abc; Domain=.platform.agpt.co; Domain=attacker.test; Secure; HttpOnly",
                "session=abc; Path=/; HttpOnly",
                "session=abc; Path=/; Secure",
                "session=abc\r\nSet-Cookie: other=value; Secure; HttpOnly",
            )
            .forEach { assertNull(it, SessionCookie.forOrigin(it, origin)) }
    }

    @Test
    fun requiresOneUsableSessionTokenBeforeReplacingAnAccount() {
        val now = 1_700_000_000_000L
        assertTrue(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_token=token; Path=/; HttpOnly; Max-Age=60"),
                now,
            )
        )
        assertTrue(
            SessionCookie.hasUsableSessionToken(
                listOf("__Secure-better-auth.session_token=token; Path=/; Secure; HttpOnly"),
                now,
            )
        )
        assertFalse(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_data=cache; Path=/; HttpOnly"),
                now,
            )
        )
        assertFalse(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_token=token; Max-Age=0"),
                now,
            )
        )
        assertFalse(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_token=token; Max-Age=invalid"),
                now,
            )
        )
        assertFalse(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_token=token; Expires=Wed, 01 Jan 2020 00:00:00 GMT"),
                now,
            )
        )
        assertFalse(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_token=a", "better-auth.session_token=b"),
                now,
            )
        )
        assertFalse(
            SessionCookie.hasUsableSessionToken(
                listOf("better-auth.session_token=a", "__Secure-better-auth.session_token=b"),
                now,
            )
        )
        assertTrue(
            SessionCookie.hasUsableSessionToken(
                listOf(
                    "better-auth.session_token=token; Max-Age=60; Expires=Wed, 01 Jan 2020 00:00:00 GMT"
                ),
                now,
            )
        )
    }

    @Test
    fun debugHttpDoesNotRequireSecureButAlwaysRequiresHttpOnly() {
        val local = ServerOrigin.parse("http://10.0.2.2:8765", true)!!
        assertNotNull(SessionCookie.forOrigin("session=abc; Path=/; HttpOnly; SameSite=Lax", local))
        assertNull(SessionCookie.forOrigin("session=abc; Path=/", local))
    }
}
