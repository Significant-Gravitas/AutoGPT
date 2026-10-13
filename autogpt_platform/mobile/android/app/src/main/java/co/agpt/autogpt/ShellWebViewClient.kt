package co.agpt.autogpt

import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient

class ShellWebViewClient(private val activity: MainActivity) : WebViewClient() {
    private var failed = false

    override fun onPageStarted(view: WebView, url: String, favicon: android.graphics.Bitmap?) { failed = false }
    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
        if (!request.isForMainFrame) return false
        val url = request.url.toString()
        if (activity.policy.isLogin(url)) { activity.signIn(); return true }
        if (activity.policy.isTrusted(url)) return false
        activity.openBrowser(url)
        return true
    }

    override fun onPageFinished(view: WebView, url: String) {
        if (!failed && !url.contains("/api/auth/mobile/")) activity.showMessage(null)
    }

    override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
        if (request.isForMainFrame) {
            failed = true
            activity.showMessage("Could not connect. Check your connection and tap Refresh.")
        }
    }

    override fun onReceivedHttpError(view: WebView, request: WebResourceRequest, errorResponse: WebResourceResponse) {
        if (request.isForMainFrame) {
            failed = true
            activity.showMessage("AutoGPT could not load this page. Tap Sign in or Refresh to try again.")
        }
    }
}
