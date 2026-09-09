package com.agpt.mobile

import java.net.URI
import java.net.URLDecoder
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64

data class AuthSession(
    val origin: ServerOrigin,
    val verifier: String,
    val state: String,
    val startedAt: Long,
) {
    val startUrl: String
        get() =
            "${origin.value}/api/auth/mobile/start?code_challenge=${challenge(verifier)}&state=$state"

    sealed interface Callback {
        data class Code(val value: String) : Callback

        data object Canceled : Callback
    }

    fun callbackCode(value: String, now: Long): String? =
        (callback(value, now) as? Callback.Code)?.value

    fun callback(value: String, now: Long): Callback? = runCatching {
        if (now < startedAt || now - startedAt > 600_000) return null
        val uri = URI(value)
        if (
            uri.scheme != "autogpt" ||
                uri.host != "auth" ||
                uri.rawPath != "/callback" ||
                uri.port != -1 ||
                uri.rawUserInfo != null ||
                uri.rawFragment != null
        )
            return null
        val values = mutableMapOf<String, String>()
        for (part in (uri.rawQuery ?: return null).split('&')) {
            val pair = part.split('=', limit = 2)
            if (pair.size != 2) return null
            val key = URLDecoder.decode(pair[0], Charsets.UTF_8.name())
            val content = URLDecoder.decode(pair[1], Charsets.UTF_8.name())
            if (values.put(key, content) != null) return null
        }
        val returnedState = values["state"] ?: return null
        if (!MessageDigest.isEqual(state.toByteArray(), returnedState.toByteArray())) return null
        when (values.keys) {
            setOf("state", "error") ->
                if (values["error"] == "access_denied") Callback.Canceled else null
            setOf("state", "code") ->
                values["code"]
                    ?.takeIf { it.matches(Regex("[A-Za-z0-9_-]{1,2048}")) }
                    ?.let { Callback.Code(it) }
            else -> null
        }
    }
        .getOrNull()

    companion object {
        fun create(origin: ServerOrigin, now: Long): AuthSession =
            AuthSession(origin, randomToken(), randomToken(), now)

        fun challenge(verifier: String): String =
            encode(
                MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray(Charsets.US_ASCII))
            )

        private fun randomToken(): String =
            encode(ByteArray(32).also { SecureRandom().nextBytes(it) })

        private fun encode(value: ByteArray): String =
            Base64.getUrlEncoder().withoutPadding().encodeToString(value)
    }
}
