package co.agpt.autogpt

import android.net.Uri
import android.os.Message
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient

class ShellChromeClient(private val activity: MainActivity) : WebChromeClient() {
    override fun onProgressChanged(view: WebView, newProgress: Int) { activity.showProgress(newProgress) }

    override fun onShowFileChooser(view: WebView, callback: ValueCallback<Array<Uri>>, params: FileChooserParams): Boolean {
        if (!activity.policy.isTrusted(view.url.orEmpty())) { callback.onReceiveValue(null); return true }
        return activity.chooseFile(callback, params)
    }

    override fun onPermissionRequest(request: PermissionRequest) { activity.requestMicrophone(request) }
    override fun onPermissionRequestCanceled(request: PermissionRequest) { activity.cancelMicrophone(request) }

    override fun onCreateWindow(view: WebView, isDialog: Boolean, isUserGesture: Boolean, resultMsg: Message): Boolean {
        if (!isUserGesture || !activity.policy.isTrusted(view.url.orEmpty())) return false
        val popup = WebView(activity)
        popup.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                activity.openBrowser(request.url.toString())
                view.post { view.destroy() }
                return true
            }
        }
        (resultMsg.obj as WebView.WebViewTransport).webView = popup
        resultMsg.sendToTarget()
        return true
    }
}
