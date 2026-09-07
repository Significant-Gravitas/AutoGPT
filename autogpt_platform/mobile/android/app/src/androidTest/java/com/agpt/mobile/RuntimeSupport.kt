package com.agpt.mobile

import android.app.Instrumentation
import android.webkit.WebView
import android.webkit.WebViewClient
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

class RuntimeSupport(private val instrumentation: Instrumentation) {
    fun <T> main(block: () -> T): T {
        var result: Result<T>? = null
        instrumentation.runOnMainSync { result = runCatching(block) }
        return requireNotNull(result).getOrThrow()
    }

    fun <T> callback(block: (CompletableFuture<T>) -> Unit): T {
        val result = CompletableFuture<T>()
        main { block(result) }
        return result.get(35, TimeUnit.SECONDS)
    }

    fun script(view: WebView, script: String): String = callback { result ->
        view.evaluateJavascript(script) { result.complete(it) }
    }

    fun load(
        view: WebView,
        origin: String,
        html: String = "<html><body>Disposable native runtime probe</body></html>",
    ) {
        callback<Unit> { result ->
            view.webViewClient =
                object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) {
                        result.complete(Unit)
                    }
                }
            view.loadDataWithBaseURL(origin, html, "text/html", "UTF-8", origin)
        }
    }

    fun awaitScript(view: WebView, script: String, expected: String) {
        var actual = ""
        repeat(100) {
            actual = script(view, script)
            if (actual == expected) return
            Thread.sleep(25)
        }
        error("WebView probe $script expected $expected, received $actual")
    }
}
