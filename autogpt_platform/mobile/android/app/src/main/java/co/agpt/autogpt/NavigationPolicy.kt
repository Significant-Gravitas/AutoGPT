package co.agpt.autogpt

import java.net.URI

class NavigationPolicy(origin: String) {
    private val originURI = URI(origin)

    fun isTrusted(value: String): Boolean = runCatching {
        val url = URI(value)
        url.scheme == "https" && url.host == originURI.host && url.rawUserInfo == null &&
            effectivePort(url) == effectivePort(originURI)
    }.getOrDefault(false)

    fun isLogin(value: String): Boolean = isTrusted(value) && URI(value).path in listOf("/login", "/signup")

    private fun effectivePort(uri: URI) = if (uri.port == -1) 443 else uri.port
}
