package com.agpt.mobile

import org.junit.Assert.*
import org.junit.Test

class AuthSessionTest {
    private val origin = ServerOrigin.parse("https://platform.agpt.co", false)!!

    @Test
    fun usesTheRfc7636S256Vector() {
        val verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk" // pragma: allowlist secret
        val challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM" // pragma: allowlist secret
        assertEquals(challenge, AuthSession.challenge(verifier))
    }

    @Test
    fun freshSessionsHaveIndependentHighEntropyCredentials() {
        val first = AuthSession.create(origin, 1_000)
        val second = AuthSession.create(origin, 1_000)
        assertNotEquals(first.verifier, second.verifier)
        assertNotEquals(first.state, second.state)
        assertTrue(first.verifier.matches(Regex("[A-Za-z0-9_-]{43,128}")))
        assertTrue(
            first.startUrl.contains("code_challenge=${AuthSession.challenge(first.verifier)}")
        )
        assertFalse(first.startUrl.contains(first.verifier))
    }

    @Test
    fun cancellationMustAlsoBeBoundToThePendingStateAndCallback() {
        val session = AuthSession.create(origin, 1_000)
        assertEquals(
            AuthSession.Callback.Canceled,
            session.callback(
                "autogpt://auth/callback?error=access_denied&state=${session.state}",
                2_000,
            ),
        )
        assertNull(
            session.callback("autogpt://auth/callback?error=access_denied&state=wrong", 2_000)
        )
        assertNull(
            session.callback(
                "autogpt://attacker/callback?error=access_denied&state=${session.state}",
                2_000,
            )
        )
        assertNull(
            session.callback(
                "autogpt://auth/callback?error=access_denied&code=x&state=${session.state}",
                2_000,
            )
        )
    }

    @Test
    fun acceptsOnlyAnExactFreshCallbackBoundToItsState() {
        val session = AuthSession.create(origin, 1_000)
        assertEquals(
            "one-time-code",
            session.callbackCode(
                "autogpt://auth/callback?code=one-time-code&state=${session.state}",
                2_000,
            ),
        )
        listOf(
                "autogpt://auth/callback?code=x&state=wrong",
                "autogpt://attacker/callback?code=x&state=${session.state}",
                "autogpt://auth:123/callback?code=x&state=${session.state}",
                "autogpt://auth/callback/extra?code=x&state=${session.state}",
                "autogpt://auth/callback?code=x&code=y&state=${session.state}",
                "autogpt://auth/callback?code=x&state=${session.state}#fragment",
                "autogpt://user@auth/callback?code=x&state=${session.state}",
                "https://auth/callback?code=x&state=${session.state}",
            )
            .forEach { assertNull(it, session.callbackCode(it, 2_000)) }
        assertNull(
            session.callbackCode("autogpt://auth/callback?code=x&state=${session.state}", 601_001)
        )
        assertNull(
            session.callbackCode("autogpt://auth/callback?code=x&state=${session.state}", 999)
        )
    }
}
