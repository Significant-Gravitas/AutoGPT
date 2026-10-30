package co.agpt.autogpt

import org.junit.Assert.*
import org.junit.Test

class ShellPolicyTest {
    private val policy = NavigationPolicy("https://platform.agpt.co")

    @Test fun onlyExactHTTPSOriginIsTrusted() {
        assertTrue(policy.isTrusted("https://platform.agpt.co/home?sessionId=1"))
        listOf("http://platform.agpt.co", "https://platform.agpt.co.evil.com", "https://platform.agpt.co:444", "https://evil.com@platform.agpt.co", "javascript:alert(1)", "file:///etc/passwd", "not a URL").forEach {
            assertFalse(it, policy.isTrusted(it))
        }
    }

    @Test fun callbackRequiresMatchingStateAndOurHTTPSRoute() {
        val attempt = AuthAttempt()
        val code = "a".repeat(43)
        val valid = "https://platform.agpt.co/auth/mobile/callback?code=$code&state=${attempt.state}"
        assertEquals("code=$code&verifier=${attempt.verifier}", attempt.exchangeBody(valid, policy))
        listOf(
            valid + "&code=$code",
            valid.replace(attempt.state, "wrong"),
            valid.replace("https://platform.agpt.co", "autogpt://auth"),
            valid.replace("platform.agpt.co", "platform.agpt.co.evil.com"),
            valid.replace("platform.agpt.co", "platform.agpt.co:444"),
            valid.replace("https:", "http:"),
            valid.replace("/auth/mobile/callback", "/auth/mobile/callback/x"),
            "$valid#fragment",
        ).forEach { assertNull(it, attempt.exchangeBody(it, policy)) }
        assertNull(AuthAttempt(attempt.verifier, attempt.state, 0).exchangeBody(valid, policy))
    }
}
