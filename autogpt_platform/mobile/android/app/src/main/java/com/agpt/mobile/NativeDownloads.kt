package com.agpt.mobile

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.CancellationSignal
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.webkit.WebView
import android.widget.Toast
import androidx.webkit.JavaScriptReplyProxy
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONObject

class NativeDownloads(
    context: Context,
    private val webView: WebView,
    private val origin: ServerOrigin,
    private val chooseDocument: (Intent) -> Unit,
) {
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private var active: Active? = null
    private var pickerRequest: Active? = null
    private var attached = false
    private var closed = false
    private val timeout = Runnable { abort("The download timed out. Please try again.") }

    private class Active(val request: DownloadMessage.Start, val reply: JavaScriptReplyProxy) {
        var destination: Uri? = null
        @Volatile var stagedFile: File? = null
        @Volatile var transfer: DownloadTransfer? = null
        var writing = false
        var copying = false
        val cancelled = AtomicBoolean(false)
        val descriptor = AtomicReference<ParcelFileDescriptor?>()
        val copyThread = AtomicReference<Thread?>()
        val signal = CancellationSignal()

        fun checkActive() {
            if (cancelled.get()) throw CancellationException("Download cancelled")
        }
    }

    fun attach() {
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) return
        WebViewCompat.addWebMessageListener(webView, "AutoGPTDownloads", setOf(origin.value)) {
            view,
            message,
            sourceOrigin,
            isMainFrame,
            proxy ->
            if (
                closed ||
                    !DownloadMessage.trusted(origin, sourceOrigin.toString(), view.url, isMainFrame)
            )
                return@addWebMessageListener
            val raw = runCatching { message.data }.getOrNull() ?: ""
            val request = runCatching { DownloadMessage.parse(raw) }.getOrNull()
            if (request == null) {
                reply(
                    proxy,
                    "error",
                    DownloadMessage.requestId(raw) ?: "",
                    message = "Invalid download request.",
                )
                abort("Invalid download request.")
                return@addWebMessageListener
            }
            handle(request, proxy)
        }
        attached = true
    }

    private fun handle(message: DownloadMessage, proxy: JavaScriptReplyProxy) {
        when (message) {
            is DownloadMessage.Start -> {
                if (active != null || pickerRequest != null) {
                    reply(
                        proxy,
                        "error",
                        message.id,
                        message = "Finish the current download first.",
                    )
                    return
                }
                val next = Active(message, proxy)
                active = next
                pickerRequest = next
                touchTimeout()
                val intent =
                    Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = message.mimeType
                        putExtra(Intent.EXTRA_TITLE, message.filename)
                    }
                try {
                    chooseDocument(intent)
                } catch (_: Exception) {
                    pickerRequest = null
                    abort("The file picker could not be opened.")
                }
            }
            is DownloadMessage.Cancel -> {
                if (active?.request?.id == message.id) abort(null)
                else reply(proxy, "cancelled", message.id)
            }
            else -> {
                val current = active
                if (current == null || current.request.id != message.id) {
                    reply(proxy, "error", message.id, message = "No matching download is active.")
                    return
                }
                val transfer = current.transfer
                if (transfer == null || current.writing || current.copying) {
                    abort("Wait for the download acknowledgement before continuing.")
                    return
                }
                current.writing = true
                touchTimeout()
                localIo.execute {
                    val result = runCatching {
                        current.checkActive()
                        when (message) {
                            is DownloadMessage.Chunk ->
                                transfer.append(message.index, message.bytes)
                            is DownloadMessage.Finish -> transfer.finish()
                            else -> error("Invalid transfer message")
                        }
                    }
                    main.post {
                        if (active !== current || closed) return@post
                        current.writing = false
                        if (result.isFailure)
                            abort("The file could not be saved. Please try again.")
                        else if (message is DownloadMessage.Chunk)
                            reply(current.reply, "ack", current.request.id, index = message.index)
                        else copyToDestination(current)
                    }
                }
            }
        }
    }

    fun onDocumentCreated(uri: Uri?) {
        val current = pickerRequest
        pickerRequest = null
        if (current == null || active !== current || closed) return
        if (uri == null) {
            abort(null)
            return
        }
        if (
            uri.scheme != "content" ||
                uri.authority.isNullOrEmpty() ||
                uri.authority == "${app.packageName}.files"
        ) {
            abort("The file picker returned an unsupported destination.")
            return
        }
        current.destination = uri
        current.writing = true
        localIo.execute {
            val result = runCatching {
                current.checkActive()
                val directory =
                    File(app.cacheDir, "native-downloads").apply { check(isDirectory || mkdirs()) }
                val file = File.createTempFile("export-", ".partial", directory)
                current.stagedFile = file
                current.transfer = DownloadTransfer(current.request, file.outputStream())
                current.checkActive()
            }
            main.post {
                if (active !== current || closed) {
                    cleanStage(current)
                    return@post
                }
                current.writing = false
                if (result.isFailure) abort("The download could not be prepared.")
                else {
                    touchTimeout()
                    reply(current.reply, "ready", current.request.id)
                }
            }
        }
    }

    private fun copyToDestination(current: Active) {
        if (!copySlots.tryAcquire()) {
            abort("A file provider is still busy. Please try again later.")
            return
        }
        current.copying = true
        current.transfer = null
        touchTimeout()
        providerIo.execute {
            current.copyThread.set(Thread.currentThread())
            val result = runCatching {
                current.checkActive()
                val descriptor =
                    requireNotNull(
                        app.contentResolver.openFileDescriptor(
                            requireNotNull(current.destination),
                            "wt",
                            current.signal,
                        )
                    )
                current.descriptor.set(descriptor)
                ParcelFileDescriptor.AutoCloseOutputStream(descriptor).use { output ->
                    requireNotNull(current.stagedFile).inputStream().use { input ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            current.checkActive()
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                        }
                        current.checkActive()
                        output.flush()
                        descriptor.checkError()
                    }
                }
            }
            current.descriptor.getAndSet(null)?.let { runCatching { it.close() } }
            current.copyThread.set(null)
            Thread.interrupted()
            copySlots.release()
            cleanStage(current)
            main.post {
                if (active !== current || closed) return@post
                if (result.isFailure) abort("Saving failed. The selected file may be incomplete.")
                else {
                    active = null
                    main.removeCallbacks(timeout)
                    reply(current.reply, "complete", current.request.id)
                }
            }
        }
    }

    fun abort(message: String? = "The page changed before the download finished.") {
        main.removeCallbacks(timeout)
        val current = active ?: return
        active = null
        current.cancelled.set(true)
        if (message == null) reply(current.reply, "cancelled", current.request.id)
        else reply(current.reply, "error", current.request.id, message = message)
        if (current.copying) {
            current.copyThread.get()?.interrupt()
            cancellationIo.execute {
                current.descriptor.getAndSet(null)?.let {
                    runCatching { it.closeWithError("Download cancelled") }
                }
                runCatching { current.signal.cancel() }
            }
            Toast.makeText(app, R.string.save_interrupted, Toast.LENGTH_LONG).show()
        }
        cleanStage(current)
    }

    private fun cleanStage(current: Active) {
        localIo.execute {
            current.transfer?.abort()
            current.transfer = null
            current.stagedFile?.delete()
        }
    }

    fun close(removeListener: Boolean = true) {
        abort()
        closed = true
        if (
            removeListener &&
                attached &&
                WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)
        ) {
            WebViewCompat.removeWebMessageListener(webView, "AutoGPTDownloads")
        }
        attached = false
    }

    private fun touchTimeout() {
        main.removeCallbacks(timeout)
        main.postDelayed(timeout, 120_000)
    }

    private fun reply(
        proxy: JavaScriptReplyProxy,
        type: String,
        id: String,
        index: Int? = null,
        message: String? = null,
    ) {
        val json = JSONObject().put("type", type).put("id", id)
        index?.let { json.put("index", it) }
        message?.let { json.put("message", it) }
        if (WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER))
            runCatching { proxy.postMessage(json.toString()) }
    }

    companion object {
        private val localIo = Executors.newSingleThreadExecutor()
        private val providerIo = Executors.newFixedThreadPool(2)
        private val cancellationIo = Executors.newFixedThreadPool(2)
        private val copySlots = Semaphore(2)
    }
}
