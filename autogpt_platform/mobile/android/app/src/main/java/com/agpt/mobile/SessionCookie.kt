package com.agpt.mobile

import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

object SessionCookie {
    fun hasUsableSessionToken(headers: List<String>, nowMillis: Long): Boolean {
        val tokens = headers.filter {
            it.substringBefore('=') in
                setOf("better-auth.session_token", "__Secure-better-auth.session_token")
        }
        if (tokens.size != 1) return false
        val parts = tokens.single().split(';').map { it.trim() }
        if (parts.first().substringAfter('=', "").isEmpty()) return false
        val attributes =
            parts.drop(1).associate {
                val pair = it.split('=', limit = 2)
                pair.first().lowercase(Locale.ROOT) to pair.getOrElse(1) { "" }
            }
        attributes["max-age"]?.let {
            return (it.toLongOrNull() ?: return false) > 0
        }
        attributes["expires"]?.let {
            return runCatching {
                    ZonedDateTime.parse(it, DateTimeFormatter.RFC_1123_DATE_TIME)
                        .toInstant()
                        .toEpochMilli() > nowMillis
                }
                .getOrDefault(false)
        }
        return true
    }

    fun forOrigin(header: String, origin: ServerOrigin): String? {
        if (header.length > 8192 || header.any { it.isISOControl() }) return null
        val parts = header.split(';').map { it.trim() }
        val pair = parts.first().split('=', limit = 2)
        if (pair.size != 2 || !pair[0].matches(Regex("[!#$%&'*+.^_`|~0-9A-Za-z-]+"))) return null
        if (
            pair[1].isEmpty() ||
                pair[1].any { it == ',' || it == '"' || it == '\\' || it.isWhitespace() }
        )
            return null
        var httpOnly = false
        var secure = false
        val seen = mutableSetOf<String>()
        val accepted = mutableListOf(parts.first())
        for (part in parts.drop(1)) {
            val attribute = part.split('=', limit = 2)
            val name = attribute[0].lowercase(Locale.ROOT)
            if (!seen.add(name)) return null
            when (name) {
                "domain" -> {
                    val domain = attribute.getOrNull(1)?.lowercase(Locale.ROOT)?.removePrefix(".")
                    if (domain != origin.host) return null
                    continue
                }
                "path" -> if (attribute.getOrNull(1) != "/") return null
                "httponly" -> {
                    if (attribute.size != 1) return null
                    httpOnly = true
                }
                "secure" -> {
                    if (attribute.size != 1) return null
                    secure = true
                }
            }
            accepted.add(part)
        }
        if (!httpOnly || (origin.scheme == "https" && !secure) || "path" !in seen) return null
        return accepted.joinToString("; ")
    }
}
