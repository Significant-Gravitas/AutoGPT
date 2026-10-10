package com.agpt.mobile

import java.net.URI
import java.util.Locale

@ConsistentCopyVisibility
data class ServerOrigin
private constructor(val value: String, val scheme: String, val host: String, val port: Int) {
    val chatUrl: String
        get() = "$value/copilot"

    fun contains(url: String): Boolean {
        val uri = validUri(url) ?: return false
        return uri.scheme.lowercase(Locale.ROOT) == scheme &&
            uri.host.lowercase(Locale.ROOT) == host &&
            effectivePort(uri) == port
    }

    fun allowsEmbeddedUrl(url: String): Boolean =
        contains(url) ||
            isExternalWebUrl(url) ||
            url in setOf("about:blank", "about:srcdoc") ||
            (url.startsWith("blob:") && contains(url.removePrefix("blob:")))

    fun isLogin(url: String): Boolean {
        if (!contains(url)) return false
        return validUri(url)?.path?.trimEnd('/') == "/login"
    }

    companion object {
        const val DEFAULT = "https://platform.agpt.co"
        private val debugHosts = setOf("localhost", "127.0.0.1", "10.0.2.2")

        fun parse(value: String, debug: Boolean): ServerOrigin? {
            val uri = validUri(value.trim()) ?: return null
            if (uri.rawQuery != null || uri.rawFragment != null || uri.path !in listOf("", "/"))
                return null
            val scheme = uri.scheme.lowercase(Locale.ROOT)
            val host = uri.host.lowercase(Locale.ROOT)
            if (scheme != "https" && !(debug && scheme == "http" && host in debugHosts)) return null
            val port = effectivePort(uri)
            val suffix = if (port == defaultPort(scheme)) "" else ":$port"
            return ServerOrigin("$scheme://$host$suffix", scheme, host, port)
        }

        fun isExternalWebUrl(value: String): Boolean =
            validUri(value)?.scheme?.lowercase(Locale.ROOT) == "https"

        private fun validUri(value: String): URI? = runCatching {
            if (value.any { it.isISOControl() } || '\\' in value) return null
            val uri = URI(value)
            if (
                !uri.isAbsolute ||
                    uri.isOpaque ||
                    uri.rawUserInfo != null ||
                    uri.host.isNullOrBlank()
            )
                return null
            if (uri.host.endsWith('.') || uri.rawAuthority.endsWith(':')) return null
            if (uri.port != -1 && uri.port !in 1..65535) return null
            uri
        }
            .getOrNull()

        private fun effectivePort(uri: URI) =
            if (uri.port == -1) defaultPort(uri.scheme.lowercase(Locale.ROOT)) else uri.port

        private fun defaultPort(scheme: String) =
            if (scheme == "https") 443 else if (scheme == "http") 80 else -1
    }
}
