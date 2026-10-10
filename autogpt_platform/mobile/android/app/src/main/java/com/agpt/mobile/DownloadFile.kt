package com.agpt.mobile

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.webkit.CookieManager
import android.webkit.URLUtil
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean

object DownloadFile {
    private const val MAX_SIZE = 50L * 1024 * 1024
    private val executor = Executors.newFixedThreadPool(2)

    fun safeName(url: String, disposition: String?, mimeType: String?): String =
        URLUtil.guessFileName(url, disposition, mimeType)
            .replace(Regex("[^A-Za-z0-9._ -]"), "_")
            .trim(' ', '.')
            .take(120)
            .ifEmpty { "download" }

    fun fetch(
        context: Context,
        origin: ServerOrigin,
        url: String,
        filename: String,
        done: (CancellableRequest, Result<File>) -> Unit,
    ): CancellableRequest {
        val app = context.applicationContext
        val main = Handler(Looper.getMainLooper())
        val delivered = AtomicBoolean(false)
        val cookie = CookieManager.getInstance().getCookie(url)
        val userAgent = android.webkit.WebSettings.getDefaultUserAgent(app)
        val request =
            CancellableRequest(300_000) { timeout ->
                main.post {
                    if (!timeout.cancelledByCaller && delivered.compareAndSet(false, true))
                        done(timeout, Result.failure(TimeoutException("Download timed out")))
                }
            }
        val future = executor.submit {
            var directory: File? = null
            val result = runCatching {
                request.checkActive()
                require(origin.contains(url))
                cleanOldFiles(app.cacheDir)
                directory =
                    File(app.cacheDir, "downloads/${UUID.randomUUID()}").apply { check(mkdirs()) }
                val output = File(requireNotNull(directory), filename)
                var current = url
                var completed = false
                repeat(6) {
                    if (completed) return@repeat
                    request.checkActive()
                    val connection = URL(current).openConnection() as HttpURLConnection
                    request.attach(connection)
                    try {
                        connection.instanceFollowRedirects = false
                        connection.connectTimeout = 20_000
                        connection.readTimeout = 30_000
                        connection.setRequestProperty("User-Agent", userAgent)
                        cookie?.let { connection.setRequestProperty("Cookie", it) }
                        if (connection.responseCode in listOf(301, 302, 303, 307, 308)) {
                            current =
                                URL(
                                        URL(current),
                                        connection.getHeaderField("Location")
                                            ?: error("Missing location"),
                                    )
                                    .toString()
                            check(origin.contains(current))
                        } else {
                            check(connection.responseCode == HttpURLConnection.HTTP_OK)
                            check(connection.contentLengthLong <= MAX_SIZE)
                            connection.inputStream.use { input ->
                                output.outputStream().use { sink ->
                                    val buffer = ByteArray(16 * 1024)
                                    var total = 0L
                                    while (true) {
                                        request.checkActive()
                                        val count = input.read(buffer)
                                        if (count < 0) break
                                        total += count
                                        check(total <= MAX_SIZE)
                                        sink.write(buffer, 0, count)
                                    }
                                }
                            }
                            completed = true
                        }
                    } finally {
                        request.detach(connection)
                        connection.disconnect()
                    }
                }
                request.checkActive()
                check(completed)
                output
            }
            val completed = request.complete()
            if (result.isFailure || !completed || request.cancelledByCaller)
                directory?.deleteRecursively()
            if (!completed) return@submit
            main.post {
                if (request.cancelledByCaller) {
                    directory?.let { executor.execute { it.deleteRecursively() } }
                } else if (delivered.compareAndSet(false, true)) done(request, result)
            }
        }
        request.attach(future)
        return request
    }

    fun clearExports(cache: File) {
        val existing = File(cache, "downloads").listFiles()?.toList().orEmpty()
        executor.execute { existing.forEach { it.deleteRecursively() } }
    }

    fun cleanOldFiles(cache: File) {
        val root = File(cache, "downloads")
        val files = root.listFiles()?.sortedByDescending { it.lastModified() } ?: return
        var bytes = 0L
        for (directory in files) {
            bytes += directory.walkTopDown().filter { it.isFile }.sumOf { it.length() }
            if (
                System.currentTimeMillis() - directory.lastModified() > 86_400_000 ||
                    bytes > 2 * MAX_SIZE
            )
                directory.deleteRecursively()
        }
    }
}
