package co.agpt.autogpt

import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun exchangeAuthCode(origin: String, body: String): List<String> = withContext(Dispatchers.IO) {
    val connection = URL("$origin/api/auth/mobile/exchange").openConnection() as HttpURLConnection
    try {
        connection.instanceFollowRedirects = false
        connection.requestMethod = "POST"
        connection.connectTimeout = 15_000
        connection.readTimeout = 15_000
        connection.doOutput = true
        connection.setRequestProperty("Origin", origin)
        connection.setRequestProperty("User-Agent", "AutoGPTMobile/0.1 Android")
        connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
        connection.outputStream.use { it.write(body.toByteArray()) }
        check(connection.responseCode == 302 && connection.getHeaderField("Location") == "$origin/home")
        val cookies = connection.headerFields.entries
            .filter { it.key.equals("Set-Cookie", ignoreCase = true) }
            .flatMap { it.value }
        check(cookies.isNotEmpty())
        cookies
    } finally { connection.disconnect() }
}
