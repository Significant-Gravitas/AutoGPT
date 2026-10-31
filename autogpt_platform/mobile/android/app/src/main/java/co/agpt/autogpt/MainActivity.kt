package co.agpt.autogpt

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.browser.customtabs.CustomTabsIntent
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    val policy = NavigationPolicy(BuildConfig.APP_ORIGIN)
    lateinit var webView: WebView
    private lateinit var message: TextView
    private lateinit var progress: ProgressBar
    private var attempt: AuthAttempt? = null
    private var upload: ValueCallback<Array<Uri>>? = null
    private var microphone: PermissionRequest? = null

    private val filePicker = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        upload?.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data))
        upload = null
    }
    private val recordAudio = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        microphone?.let {
            if (granted && policy.isTrusted(it.origin.toString()) && policy.isTrusted(webView.url.orEmpty()))
                it.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
            else it.deny()
        }
        microphone = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val bar = LinearLayout(this)
        fun button(label: String, action: () -> Unit) {
            bar.addView(Button(this).apply { text = label; setOnClickListener { action() } }, LinearLayout.LayoutParams(0, 48.dp(), 1f))
        }
        button("Back") { if (webView.canGoBack()) webView.goBack() }
        button("Home") { goHome() }
        button("Refresh") { if (webView.url?.contains("/api/auth/mobile/") == false) webView.reload() else goHome() }
        button("Sign in") { signIn() }
        message = TextView(this).apply { visibility = View.GONE; setPadding(16.dp(), 12.dp(), 16.dp(), 12.dp()) }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
        webView = WebView(this)
        root.addView(bar)
        root.addView(progress, LinearLayout.LayoutParams(-1, 3.dp()))
        root.addView(message)
        root.addView(webView, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val padding = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(padding.left, padding.top, padding.right, padding.bottom)
            insets
        }
        configureWebView()
        val verifier = savedInstanceState?.getString("authVerifier")
        val state = savedInstanceState?.getString("authState")
        if (verifier != null && state != null) attempt = AuthAttempt(verifier, state, savedInstanceState.getLong("authCreatedAt"))
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { if (webView.canGoBack()) webView.goBack() else finish() }
        })
        if (savedInstanceState == null || webView.restoreState(savedInstanceState) == null) goHome()
        handleAuthCallback(intent)
    }

    @Suppress("SetJavaScriptEnabled")
    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = true
            mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
            setSupportMultipleWindows(true)
            userAgentString += " AutoGPTMobile/0.1 Android"
        }
        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, false)
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
        webView.webViewClient = ShellWebViewClient(this)
        webView.webChromeClient = ShellChromeClient(this)
        webView.setDownloadListener { url, _, _, _, _ ->
            if (Uri.parse(url).scheme in listOf("https", "http")) {
                openBrowser(url)
                showMessage("Download opened in your browser. You may need to sign in there.")
            } else showMessage("Open AutoGPT in your browser to save this file.")
        }
    }

    fun signIn() {
        attempt = AuthAttempt()
        openBrowser(attempt!!.startURL(BuildConfig.APP_ORIGIN))
        showMessage("Finish signing in in your browser. Tap Sign in to start again.")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleAuthCallback(intent)
    }

    private fun handleAuthCallback(intent: Intent) {
        val uri = intent.data ?: return
        intent.data = null
        val body = attempt?.exchangeBody(uri.toString())
        if (body == null) { showMessage("Sign-in expired or did not match this device. Tap Sign in to try again."); return }
        attempt = null
        lifecycleScope.launch {
            try {
                val cookies = exchangeAuthCode(BuildConfig.APP_ORIGIN, body)
                var remaining = cookies.size
                for (cookie in cookies) {
                    CookieManager.getInstance().setCookie(BuildConfig.APP_ORIGIN, cookie) {
                        remaining -= 1
                        if (remaining == 0 && !isDestroyed) {
                            CookieManager.getInstance().flush()
                            goHome()
                        }
                    }
                }
            } catch (_: Exception) {
                showMessage("Sign-in could not finish. Tap Sign in to try again.")
            }
        }
    }

    fun openBrowser(value: String) {
        val uri = Uri.parse(value)
        if (uri.scheme !in listOf("https", "http")) return
        try { CustomTabsIntent.Builder().build().launchUrl(this, uri) }
        catch (_: ActivityNotFoundException) { showMessage("Install a browser to open this link.") }
    }

    fun chooseFile(callback: ValueCallback<Array<Uri>>, params: WebChromeClient.FileChooserParams): Boolean {
        upload?.onReceiveValue(null)
        upload = callback
        return try { filePicker.launch(params.createIntent()); true }
        catch (_: ActivityNotFoundException) { upload = null; callback.onReceiveValue(null); true }
    }

    fun requestMicrophone(request: PermissionRequest) {
        if (!policy.isTrusted(request.origin.toString()) || request.resources.toList() != listOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE)) {
            request.deny(); return
        }
        microphone?.deny()
        microphone = request
        recordAudio.launch(Manifest.permission.RECORD_AUDIO)
    }

    fun cancelMicrophone(request: PermissionRequest) { if (microphone === request) microphone = null }
    fun goHome() { webView.loadUrl("${BuildConfig.APP_ORIGIN}/home") }
    fun showMessage(text: String?) { message.text = text; message.visibility = if (text == null) View.GONE else View.VISIBLE }
    fun showProgress(value: Int) { progress.progress = value; progress.visibility = if (value == 100) View.GONE else View.VISIBLE }
    private fun Int.dp() = (this * resources.displayMetrics.density).toInt()

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        webView.saveState(outState)
        attempt?.let {
            outState.putString("authVerifier", it.verifier)
            outState.putString("authState", it.state)
            outState.putLong("authCreatedAt", it.createdAt)
        }
    }

    override fun onPause() { CookieManager.getInstance().flush(); super.onPause() }
    override fun onDestroy() {
        upload?.onReceiveValue(null)
        microphone?.deny()
        webView.destroy()
        super.onDestroy()
    }
}
