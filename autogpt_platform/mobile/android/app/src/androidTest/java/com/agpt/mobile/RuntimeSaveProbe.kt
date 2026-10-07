package com.agpt.mobile

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.provider.MediaStore
import android.webkit.WebView
import java.util.Base64
import java.util.UUID
import org.json.JSONObject

class RuntimeSaveProbe(private val context: Context, private val runtime: RuntimeSupport) {
    @SuppressLint("SetJavaScriptEnabled")
    fun run() {
        val resolver = context.contentResolver
        val filename = "autogpt-runtime-${UUID.randomUUID()}.bin"
        val destination =
            requireNotNull(
                resolver.insert(
                    MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                    ContentValues().apply {
                        put(MediaStore.Downloads.DISPLAY_NAME, filename)
                        put(MediaStore.Downloads.MIME_TYPE, "application/octet-stream")
                        put(MediaStore.Downloads.RELATIVE_PATH, "Download/AutoGPTRuntimeProbe")
                        put(MediaStore.Downloads.IS_PENDING, 1)
                    },
                )
            )
        val origin = requireNotNull(ServerOrigin.parse("https://android-runtime.invalid", false))
        var view: WebView? = null
        var downloads: NativeDownloads? = null
        try {
            val browser = runtime.main {
                WebView(context).apply { settings.javaScriptEnabled = true }
            }
            view = browser
            runtime.main {
                downloads =
                    NativeDownloads(context, browser, origin) {
                        downloads?.onDocumentCreated(destination)
                    }
                downloads?.attach()
            }
            runtime.load(browser, origin.chatUrl)
            runtime.script(
                browser,
                "AutoGPTDownloads.onmessage = event => { window.saveReply = JSON.parse(event.data); };",
            )
            val expected = ByteArray(96 * 1024 + 17) { (it % 251).toByte() }
            send(
                browser,
                JSONObject()
                    .put("type", "start")
                    .put("id", "save")
                    .put("filename", filename)
                    .put("mimeType", "application/octet-stream")
                    .put("size", expected.size),
            )
            awaitReply(browser, "ready", "save")
            expected.asList().chunked(48 * 1024).forEachIndexed { index, chunk ->
                send(
                    browser,
                    JSONObject()
                        .put("type", "chunk")
                        .put("id", "save")
                        .put("index", index)
                        .put("data", Base64.getEncoder().encodeToString(chunk.toByteArray())),
                )
                awaitReply(browser, "ack", "save")
                check(runtime.script(browser, "window.saveReply.index") == index.toString())
            }
            send(browser, JSONObject().put("type", "finish").put("id", "save"))
            awaitReply(browser, "complete", "save")
            check(
                requireNotNull(resolver.openInputStream(destination))
                    .use { it.readBytes() }
                    .contentEquals(expected)
            ) {
                "Provider bytes differ from the acknowledged download"
            }
            send(
                browser,
                JSONObject()
                    .put("type", "start")
                    .put("id", "cancel")
                    .put("filename", filename)
                    .put("mimeType", "application/octet-stream")
                    .put("size", 3),
            )
            awaitReply(browser, "ready", "cancel")
            send(
                browser,
                JSONObject()
                    .put("type", "chunk")
                    .put("id", "cancel")
                    .put("index", 0)
                    .put("data", "YWJj"),
            )
            awaitReply(browser, "ack", "cancel")
            send(browser, JSONObject().put("type", "cancel").put("id", "cancel"))
            awaitReply(browser, "cancelled", "cancel")
            check(
                requireNotNull(resolver.openInputStream(destination))
                    .use { it.readBytes() }
                    .contentEquals(expected)
            ) {
                "Cancellation modified or removed the existing test-owned destination"
            }
        } finally {
            runtime.main {
                downloads?.close()
                view?.destroy()
            }
            check(resolver.delete(destination, null, null) == 1) {
                "Could not remove the probe-owned MediaStore row"
            }
            resolver.query(destination, arrayOf(MediaStore.Downloads._ID), null, null, null)?.use {
                cursor ->
                check(!cursor.moveToFirst()) { "Probe-owned MediaStore row still exists" }
            }
        }
    }

    private fun send(view: WebView, message: JSONObject) {
        runtime.script(
            view,
            "window.saveReply = null; AutoGPTDownloads.postMessage(${JSONObject.quote(message.toString())});",
        )
    }

    private fun awaitReply(view: WebView, type: String, id: String) {
        runtime.awaitScript(
            view,
            "window.saveReply && window.saveReply.type",
            JSONObject.quote(type),
        )
        check(runtime.script(view, "window.saveReply.id") == JSONObject.quote(id))
    }
}
