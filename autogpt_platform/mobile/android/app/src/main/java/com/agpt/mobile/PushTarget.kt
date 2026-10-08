package com.agpt.mobile

import java.net.URI
import java.net.URLDecoder

object PushTarget {
    fun url(
        path: String,
        notificationOrigin: String,
        binding: String,
        expectedBinding: String,
        origin: ServerOrigin,
    ): String? = runCatching {
        if (binding.isEmpty() || binding != expectedBinding || notificationOrigin != origin.value)
            return null
        if (!path.startsWith("/") || path.startsWith("//")) return null
        val uri = URI(path)
        if (uri.isAbsolute || uri.host != null || uri.fragment != null) return null
        val query = uri.rawQuery ?: return null
        val pair = query.split("=", limit = 2)
        if (pair.size != 2 || '&' in query) return null
        val value = URLDecoder.decode(pair[1], Charsets.UTF_8.name())
        if (value.isEmpty() || value.length > 128) return null
        if (
            !(uri.path == "/home" && pair[0] == "sessionId") &&
                !(uri.path == "/mobile" && pair[0] == "tab" && value == "attention")
        )
            return null
        (origin.value + path).takeIf { origin.contains(it) }
    }
        .getOrNull()
}
