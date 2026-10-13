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

    @Test fun callbackRequiresMatchingStateAndExactRoute() {
        val attempt = AuthAttempt()
        val code = "a".repeat(43)
        val valid = "autogpt://auth/callback?code=$code&state=${attempt.state}"
        assertEquals("code=$code&verifier=${attempt.verifier}", attempt.exchangeBody(valid))
        assertNull(attempt.exchangeBody(valid + "&code=$code"))
        assertNull(attempt.exchangeBody(valid.replace(attempt.state, "wrong")))
        assertNull(attempt.exchangeBody(valid.replace("//auth", "//evil")))
        assertNull(attempt.exchangeBody(valid.replace("autogpt:", "https:")))
        assertNull(AuthAttempt(attempt.verifier, attempt.state, 0).exchangeBody(valid))
    }
}
