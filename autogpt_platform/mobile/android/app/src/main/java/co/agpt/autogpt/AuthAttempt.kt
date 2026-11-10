package co.agpt.autogpt

import java.net.URI
import java.net.URLDecoder
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64

class AuthAttempt(
    val verifier: String = randomToken(),
    val state: String = randomToken(),
    val createdAt: Long = System.currentTimeMillis(),
) {
    fun startURL(origin: String): String {
        val challenge = encode(MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray()))
        return "$origin/api/auth/mobile/start?challenge=$challenge&state=$state"
    }

    fun exchangeBody(callback: String): String? = runCatching {
        val uri = URI(callback)
        if (System.currentTimeMillis() - createdAt >= 600_000 || uri.scheme != "autogpt" ||
            uri.host != "auth" || uri.path != "/callback" || uri.rawUserInfo != null ||
            uri.port != -1 || uri.fragment != null) return null
        val items = uri.rawQuery?.split("&")?.map {
            val pair = it.split("=", limit = 2)
            URLDecoder.decode(pair[0], "UTF-8") to URLDecoder.decode(pair.getOrElse(1) { "" }, "UTF-8")
        } ?: return null
        if (items.size != 2 || items.count { it.first == "state" } != 1 ||
            items.firstOrNull { it.first == "state" }?.second != state) return null
        val code = items.firstOrNull { it.first == "code" }?.second ?: return null
        if (!code.matches(Regex("^[A-Za-z0-9_-]{43}$"))) return null
        "code=$code&verifier=$verifier"
    }.getOrNull()

    companion object {
        private fun randomToken(): String = encode(ByteArray(32).also { SecureRandom().nextBytes(it) })
        private fun encode(bytes: ByteArray): String = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    }
}
