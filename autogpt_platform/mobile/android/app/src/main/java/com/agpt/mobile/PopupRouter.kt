package com.agpt.mobile

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.Message
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient

class PopupRouter(private val context: Context, private val open: (String) -> Unit) {
    private val popups = mutableSetOf<WebView>()
    private val main = Handler(Looper.getMainLooper())

    fun create(message: Message, userGesture: Boolean): Boolean {
        if (!userGesture || popups.isNotEmpty()) return false
        val transport = message.obj as? WebView.WebViewTransport ?: return false
        val popup = WebView(context)
        popups.add(popup)
        popup.settings.apply {
            javaScriptEnabled = false
            domStorageEnabled = false
            allowFileAccess = false
            allowContentAccess = false
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            blockNetworkLoads = true
        }
        var routed = false
        fun route(url: String) {
            if (routed || url == "about:blank") return
            routed = true
            main.post {
                if (popups.contains(popup)) {
                    close(popup)
                    open(url)
                }
            }
        }
        popup.webViewClient =
            object : WebViewClient() {
                override fun shouldOverrideUrlLoading(
                    view: WebView,
                    request: WebResourceRequest,
                ): Boolean {
                    route(request.url.toString())
                    return request.url.toString() != "about:blank"
                }

                override fun onPageStarted(
                    view: WebView,
                    url: String,
                    favicon: android.graphics.Bitmap?,
                ) {
                    route(url)
                }

                override fun onRenderProcessGone(
                    view: WebView,
                    detail: RenderProcessGoneDetail,
                ): Boolean {
                    popups.remove(view)
                    view.destroy()
                    return true
                }
            }
        transport.webView = popup
        message.sendToTarget()
        main.postDelayed({ close(popup) }, 10_000)
        return true
    }

    fun closeAll() {
        popups.toList().forEach { close(it) }
        main.removeCallbacksAndMessages(null)
    }

    private fun close(popup: WebView) {
        if (popups.remove(popup)) {
            popup.destroy()
        }
    }
}
