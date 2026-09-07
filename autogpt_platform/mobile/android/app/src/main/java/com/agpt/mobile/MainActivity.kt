package com.agpt.mobile

import android.Manifest
import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.net.Uri
import android.net.http.SslError
import android.os.Bundle
import android.os.Parcel
import android.view.View
import android.view.inputmethod.EditorInfo
import android.webkit.CookieManager
import android.webkit.PermissionRequest
import android.webkit.RenderProcessGoneDetail
import android.webkit.SslErrorHandler
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.browser.customtabs.CustomTabsIntent
import androidx.core.content.ContextCompat
import androidx.core.graphics.Insets
import androidx.core.net.toUri
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.ViewModelProvider

class MainActivity : ComponentActivity() {
    private lateinit var layout: BrowserLayout
    private lateinit var origin: ServerOrigin
    private lateinit var auth: AuthViewModel
    private lateinit var popups: PopupRouter
    private var webView: WebView? = null
    private var documentGeneration = 0
    private val httpDownloads = mutableSetOf<CancellableRequest>()
    private var nativeDownloads: NativeDownloads? = null
    private var documentPickerOutstanding = false
    private var microphonePromptOutstanding = false

    private data class MicrophonePermission(
        val view: WebView,
        val request: PermissionRequest,
        val origin: ServerOrigin,
        val generation: Int,
    )

    private var microphonePermission: MicrophonePermission? = null

    private data class FileSelection(
        val view: WebView,
        val callback: ValueCallback<Array<Uri>>,
        val generation: Int,
        val origin: ServerOrigin,
        val multiple: Boolean,
    )

    private var fileSelection: FileSelection? = null
    private var filePickerOutstanding = false
    private var lastSafeUrl = ""
    private var pageFailed = false
    private var pendingError: Int? = null
    private var backCallback: OnBackPressedCallback? = null
    private val preferences by lazy { getSharedPreferences("server", MODE_PRIVATE) }

    private val filePicker =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            filePickerOutstanding = false
            val selection = fileSelection ?: return@registerForActivityResult
            fileSelection = null
            if (
                result.resultCode != RESULT_OK ||
                    selection.view !== webView ||
                    selection.generation != documentGeneration ||
                    selection.origin != origin ||
                    auth.replacingSession
            ) {
                selection.callback.onReceiveValue(null)
                return@registerForActivityResult
            }
            val data = result.data
            val uris =
                if (data?.clipData != null) {
                    val clip = data.clipData!!
                    if (clip.itemCount > 32) {
                        selection.callback.onReceiveValue(null)
                        toast(R.string.too_many_files)
                        return@registerForActivityResult
                    }
                    (0 until clip.itemCount).map { clip.getItemAt(it).uri }
                } else listOfNotNull(data?.data)
            val valid =
                uris.isNotEmpty() &&
                    (selection.multiple || uris.size == 1) &&
                    uris.all { uri ->
                        uri.scheme == "content" &&
                            !uri.authority.isNullOrBlank() &&
                            uri.authority != "$packageName.files"
                    }
            selection.callback.onReceiveValue(if (valid) uris.toTypedArray() else null)
            if (!valid) toast(R.string.file_selection_invalid)
        }

    private val documentPicker =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val wasExpected = documentPickerOutstanding
            documentPickerOutstanding = false
            if (!wasExpected) return@registerForActivityResult
            val uri = result.data?.data?.takeIf { result.resultCode == RESULT_OK }
            val downloads = nativeDownloads
            if (downloads != null) downloads.onDocumentCreated(uri)
        }

    private val microphonePrompt =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            microphonePromptOutstanding = false
            val pending = microphonePermission ?: return@registerForActivityResult
            microphonePermission = null
            if (
                granted &&
                    pending.view === webView &&
                    pending.origin == origin &&
                    pending.generation == documentGeneration &&
                    !auth.replacingSession &&
                    origin.contains(pending.request.origin.toString()) &&
                    origin.contains(pending.view.url ?: "")
            ) {
                pending.request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
            } else runCatching { pending.request.deny() }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        documentPickerOutstanding = savedInstanceState?.getBoolean("document_picker") == true
        filePickerOutstanding = savedInstanceState?.getBoolean("file_picker") == true
        microphonePromptOutstanding = savedInstanceState?.getBoolean("microphone_prompt") == true
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowCompat.getInsetsController(window, window.decorView).apply {
            val light =
                resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK !=
                    Configuration.UI_MODE_NIGHT_YES
            isAppearanceLightStatusBars = light
            isAppearanceLightNavigationBars = light
        }
        origin =
            ServerOrigin.parse(
                preferences.getString("origin", ServerOrigin.DEFAULT) ?: ServerOrigin.DEFAULT,
                BuildConfig.DEBUG,
            ) ?: requireNotNull(ServerOrigin.parse(ServerOrigin.DEFAULT, false))
        lastSafeUrl = origin.chatUrl
        auth = ViewModelProvider(this)[AuthViewModel::class.java]
        layout = BrowserLayout(this)
        setContentView(layout)
        popups =
            PopupRouter(this) { url ->
                when {
                    origin.isLogin(url) -> showSignIn()
                    origin.contains(url) -> load(url)
                    else -> openExternal(url)
                }
            }
        installInsets()
        layout.back.setOnClickListener { webView?.takeIf { it.canGoBack() }?.goBack() }
        layout.menu.setOnClickListener { showMenu(it) }
        backCallback =
            object : OnBackPressedCallback(false) {
                    override fun handleOnBackPressed() {
                        webView?.takeIf { it.canGoBack() }?.goBack()
                    }
                }
                .also { onBackPressedDispatcher.addCallback(this, it) }
        if (!auth.replacingSession) {
            createWebView()
            val restored =
                savedInstanceState
                    ?.takeIf { it.getString("origin") == origin.value }
                    ?.getBundle("web_state")
                    ?.let { state ->
                        val history = webView?.restoreState(state)
                        history != null &&
                            (0 until history.size).all {
                                origin.contains(history.getItemAtIndex(it).url)
                            }
                    } == true
            if (!restored) {
                val candidate = savedInstanceState?.getString("last_url")
                load(candidate?.takeIf { origin.contains(it) } ?: origin.chatUrl)
            }
        }
        if (intent?.action == Intent.ACTION_VIEW) acceptIntent(intent)
        DownloadFile.cleanOldFiles(cacheDir)
    }

    override fun onStart() {
        super.onStart()
        auth.listener = { updateAuth() }
        updateAuth()
    }

    override fun onStop() {
        auth.listener = null
        CookieManager.getInstance().flush()
        super.onStop()
    }

    override fun onResume() {
        super.onResume()
        webView?.onResume()
        pendingError?.let {
            pendingError = null
            showError(it)
        }
    }

    override fun onPause() {
        webView?.onPause()
        super.onPause()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(Intent(intent).setData(null))
        acceptIntent(intent)
    }

    private fun acceptIntent(intent: Intent) {
        if (intent.action != Intent.ACTION_VIEW) return
        val callback = intent.dataString ?: return
        intent.data = null
        auth.acceptCallback(callback, origin)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean("document_picker", documentPickerOutstanding)
        outState.putBoolean("file_picker", filePickerOutstanding)
        outState.putBoolean("microphone_prompt", microphonePromptOutstanding)
        outState.putString("origin", origin.value)
        outState.putString("last_url", lastSafeUrl)
        val state = Bundle()
        webView?.saveState(state)
        val parcel = Parcel.obtain()
        try {
            parcel.writeBundle(state)
            if (parcel.dataSize() <= 128 * 1024) outState.putBundle("web_state", state)
        } finally {
            parcel.recycle()
        }
        super.onSaveInstanceState(outState)
    }

    override fun onDestroy() {
        cancelFileSelection()
        destroyWebView()
        super.onDestroy()
    }

    private fun installInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(layout) { view, insets ->
            val types =
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            val bars = insets.getInsets(types)
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.Builder(insets).setInsets(types, Insets.NONE).build()
        }
        ViewCompat.requestApplyInsets(layout)
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun createWebView() {
        val view = WebView(this)
        webView = view
        view.layoutParams =
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            )
        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = true
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            javaScriptCanOpenWindowsAutomatically = false
            setSupportMultipleWindows(true)
            mediaPlaybackRequiresUserGesture = true
            userAgentString = "$userAgentString AutoGPTMobile/Android"
            setSupportZoom(true)
            builtInZoomControls = true
            displayZoomControls = false
            safeBrowsingEnabled = true
        }
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
        CookieManager.getInstance().apply {
            setAcceptCookie(true)
            setAcceptThirdPartyCookies(view, false)
        }
        view.webViewClient = browserClient()
        view.webChromeClient = chromeClient(view)
        view.setDownloadListener { url, _, disposition, mimeType, _ ->
            download(url, disposition, mimeType)
        }
        nativeDownloads =
            NativeDownloads(this, view, origin) { intent ->
                    check(!documentPickerOutstanding)
                    documentPickerOutstanding = true
                    try {
                        documentPicker.launch(intent)
                    } catch (error: Exception) {
                        documentPickerOutstanding = false
                        throw error
                    }
                }
                .also { it.attach() }
        layout.webContainer.addView(view)
    }

    private fun destroyWebView(rendererGone: Boolean = false) {
        cancelMicrophonePermission()
        documentGeneration++
        httpDownloads.forEach { it.cancel() }
        httpDownloads.clear()
        popups.closeAll()
        nativeDownloads?.close(removeListener = !rendererGone)
        nativeDownloads = null
        webView?.let {
            layout.webContainer.removeView(it)
            if (!rendererGone) it.stopLoading()
            it.destroy()
        }
        webView = null
        updateBack()
    }

    private fun browserClient() =
        object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView,
                request: WebResourceRequest,
            ): Boolean {
                val url = request.url.toString()
                if (!request.isForMainFrame) return !origin.allowsEmbeddedUrl(url)
                if (origin.isLogin(url)) {
                    showSignIn()
                    return true
                }
                if (origin.contains(url)) return false
                if (request.hasGesture()) openExternal(url) else toast(R.string.link_blocked)
                return true
            }

            override fun onPageStarted(view: WebView, url: String, favicon: Bitmap?) {
                documentGeneration++
                cancelMicrophonePermission()
                cancelFileSelection()
                httpDownloads.forEach { it.cancel() }
                httpDownloads.clear()
                nativeDownloads?.abort()
                if (!origin.contains(url)) {
                    view.stopLoading()
                    showError(R.string.load_error_detail)
                    return
                }
                pageFailed = false
                lastSafeUrl = url
                layout.loading(5)
                updateBack()
            }

            override fun onPageCommitVisible(view: WebView, url: String) {
                if (!pageFailed && origin.contains(url)) layout.showPage()
            }

            override fun onPageFinished(view: WebView, url: String) {
                layout.loading(100)
                if (!pageFailed && origin.contains(url)) layout.showPage()
                updateBack()
            }

            override fun doUpdateVisitedHistory(view: WebView, url: String?, isReload: Boolean) {
                if (url != null && origin.contains(url)) lastSafeUrl = url
                updateBack()
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError,
            ) {
                if (request.isForMainFrame) showError(R.string.load_error_detail)
            }

            override fun onReceivedHttpError(
                view: WebView,
                request: WebResourceRequest,
                response: WebResourceResponse,
            ) {
                if (request.isForMainFrame && response.statusCode >= 400)
                    showError(R.string.load_error_detail)
            }

            override fun onReceivedSslError(
                view: WebView,
                handler: SslErrorHandler,
                error: SslError,
            ) {
                handler.cancel()
                showError(R.string.ssl_error_detail)
            }

            override fun onRenderProcessGone(
                view: WebView,
                detail: RenderProcessGoneDetail,
            ): Boolean {
                if (view !== webView) {
                    layout.webContainer.removeView(view)
                    view.destroy()
                    return true
                }
                cancelFileSelection()
                destroyWebView(rendererGone = true)
                pendingError = R.string.renderer_error_detail
                if (!isFinishing && !isDestroyed) showError(R.string.renderer_error_detail)
                return true
            }
        }

    private fun chromeClient(owner: WebView) =
        object : WebChromeClient() {
            override fun onProgressChanged(view: WebView, newProgress: Int) {
                if (!pageFailed) layout.loading(newProgress)
            }

            override fun onPermissionRequest(request: PermissionRequest) {
                runOnUiThread { requestMicrophone(owner, request) }
            }

            override fun onPermissionRequestCanceled(request: PermissionRequest) {
                if (microphonePermission?.request === request) microphonePermission = null
            }

            override fun onCreateWindow(
                view: WebView,
                isDialog: Boolean,
                isUserGesture: Boolean,
                resultMsg: android.os.Message,
            ): Boolean = popups.create(resultMsg, isUserGesture)

            override fun onShowFileChooser(
                view: WebView,
                callback: ValueCallback<Array<Uri>>,
                params: FileChooserParams,
            ): Boolean {
                if (owner !== webView || filePickerOutstanding || auth.replacingSession) {
                    callback.onReceiveValue(null)
                    return true
                }
                filePickerOutstanding = true
                fileSelection =
                    FileSelection(
                        owner,
                        callback,
                        documentGeneration,
                        origin,
                        params.mode == FileChooserParams.MODE_OPEN_MULTIPLE,
                    )
                val types =
                    params.acceptTypes
                        .flatMap { it.split(',') }
                        .map { it.trim() }
                        .filter { it.matches(Regex("[A-Za-z0-9.+_-]+/[A-Za-z0-9.+_*-]+")) }
                        .distinct()
                        .take(32)
                val intent =
                    Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = types.singleOrNull() ?: "*/*"
                        if (types.size > 1) putExtra(Intent.EXTRA_MIME_TYPES, types.toTypedArray())
                        putExtra(
                            Intent.EXTRA_ALLOW_MULTIPLE,
                            params.mode == FileChooserParams.MODE_OPEN_MULTIPLE,
                        )
                    }
                try {
                    filePicker.launch(intent)
                } catch (_: ActivityNotFoundException) {
                    filePickerOutstanding = false
                    fileSelection = null
                    callback.onReceiveValue(null)
                    toast(R.string.file_selection_failed)
                }
                return true
            }
        }

    private fun cancelFileSelection() {
        fileSelection?.callback?.onReceiveValue(null)
        fileSelection = null
    }

    private fun cancelMicrophonePermission() {
        microphonePermission?.let { runCatching { it.request.deny() } }
        microphonePermission = null
    }

    private fun requestMicrophone(owner: WebView, request: PermissionRequest) {
        if (
            owner !== webView ||
                auth.replacingSession ||
                !origin.contains(owner.url ?: "") ||
                !origin.contains(request.origin.toString()) ||
                !request.resources.contentEquals(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
        ) {
            request.deny()
            return
        }
        if (
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        ) {
            request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
            return
        }
        if (microphonePromptOutstanding) {
            request.deny()
            return
        }
        cancelMicrophonePermission()
        microphonePermission = MicrophonePermission(owner, request, origin, documentGeneration)
        microphonePromptOutstanding = true
        microphonePrompt.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun load(url: String) {
        if (auth.replacingSession || !origin.contains(url)) return
        if (webView == null) createWebView()
        pendingError = null
        pageFailed = false
        layout.showPage()
        layout.loading(5)
        lastSafeUrl = url
        webView?.loadUrl(url)
    }

    private fun updateBack() {
        val canGoBack = webView?.canGoBack() == true
        layout.back.isEnabled = canGoBack
        layout.back.alpha = if (canGoBack) 1f else 0.35f
        backCallback?.isEnabled = canGoBack
    }

    private fun showError(detail: Int) {
        pageFailed = true
        layout.showPanel(
            R.string.unable_to_load,
            detail,
            R.string.retry,
            { load(lastSafeUrl) },
            R.string.settings to { showSettings() },
        )
    }

    private fun showSignIn() {
        pageFailed = true
        layout.showPanel(
            R.string.app_name,
            R.string.sign_in_detail,
            R.string.sign_in,
            { beginSignIn() },
            R.string.settings to { showSettings() },
        )
    }

    private fun beginSignIn() {
        if (auth.replacingSession) return
        if (!BrowserSession.supportsCompleteDeletion) {
            toast(R.string.update_webview)
            return
        }
        val url = auth.start(origin)
        if (!openBrowser(url)) auth.cancel()
    }

    private fun updateAuth() {
        layout.menu.isEnabled = !auth.replacingSession
        when (auth.status) {
            AuthViewModel.Status.WAITING -> {
                pageFailed = true
                layout.showPanel(
                    R.string.signing_in,
                    R.string.browser_sign_in_detail,
                    R.string.sign_in,
                    { beginSignIn() },
                    R.string.cancel_sign_in to
                        {
                            auth.cancel()
                            showSignIn()
                        },
                )
            }
            AuthViewModel.Status.EXCHANGING -> {
                pageFailed = true
                layout.showPanel(
                    R.string.finishing_sign_in,
                    R.string.browser_exchange_detail,
                    R.string.cancel_sign_in,
                    {
                        auth.cancel()
                        showSignIn()
                    },
                )
                layout.loading(25)
            }
            AuthViewModel.Status.READY_TO_INSTALL -> {
                cancelFileSelection()
                webView?.clearCache(true)
                destroyWebView()
                DownloadFile.clearExports(cacheDir)
                auth.installReadySession()
            }
            AuthViewModel.Status.INSTALLING,
            AuthViewModel.Status.CLEARING_SESSION -> {
                pageFailed = true
                layout.showPanel(R.string.connecting, R.string.browser_exchange_detail, 0, {})
                layout.loading(25)
            }
            AuthViewModel.Status.SESSION_CLEARED -> {
                origin =
                    ServerOrigin.parse(
                        preferences.getString("origin", ServerOrigin.DEFAULT)
                            ?: ServerOrigin.DEFAULT,
                        BuildConfig.DEBUG,
                    ) ?: requireNotNull(ServerOrigin.parse(ServerOrigin.DEFAULT, false))
                auth.acknowledge()
                load(origin.chatUrl)
                toast(R.string.session_cleared)
            }
            AuthViewModel.Status.SUCCESS -> {
                auth.acknowledge()
                webView?.clearHistory()
                load(origin.chatUrl)
                toast(R.string.signed_in)
            }
            AuthViewModel.Status.FAILED,
            AuthViewModel.Status.EXPIRED -> {
                val message =
                    if (auth.status == AuthViewModel.Status.EXPIRED) R.string.sign_in_expired
                    else R.string.sign_in_failed
                auth.acknowledge()
                pageFailed = true
                layout.showPanel(
                    R.string.sign_in,
                    message,
                    R.string.sign_in,
                    { beginSignIn() },
                    R.string.settings to { showSettings() },
                )
            }
            AuthViewModel.Status.CANCELED -> {
                auth.acknowledge()
                showSignIn()
            }
            AuthViewModel.Status.IDLE -> Unit
        }
    }

    private fun showMenu(anchor: View) {
        PopupMenu(this, anchor).apply {
            listOf(
                    R.string.new_chat,
                    R.string.sign_in,
                    R.string.reload,
                    R.string.open_browser,
                    R.string.settings,
                    R.string.clear_session,
                )
                .forEach { label ->
                    menu.add(label).setOnMenuItemClickListener {
                        if (auth.replacingSession) return@setOnMenuItemClickListener true
                        when (label) {
                            R.string.new_chat -> load(origin.chatUrl)
                            R.string.sign_in -> beginSignIn()
                            R.string.reload -> load(lastSafeUrl)
                            R.string.open_browser -> openBrowser(lastSafeUrl)
                            R.string.settings -> showSettings()
                            R.string.clear_session ->
                                AlertDialog.Builder(this@MainActivity)
                                    .setTitle(R.string.clear_session_title)
                                    .setMessage(R.string.clear_session_detail)
                                    .setNegativeButton(R.string.cancel, null)
                                    .setPositiveButton(R.string.clear) { _, _ -> clearSession() }
                                    .show()
                        }
                        true
                    }
                }
            show()
        }
    }

    private fun showSettings() {
        if (auth.replacingSession) return
        val input =
            EditText(this).apply {
                setText(origin.value)
                hint = ServerOrigin.DEFAULT
                inputType = EditorInfo.TYPE_CLASS_TEXT or EditorInfo.TYPE_TEXT_VARIATION_URI
                setSingleLine()
                selectAll()
            }
        val form =
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                val inset = (24 * resources.displayMetrics.density).toInt()
                setPadding(inset, inset / 2, inset, 0)
                addView(
                    TextView(this@MainActivity).apply {
                        text =
                            if (BuildConfig.DEBUG) {
                                getString(
                                    R.string.server_description_debug,
                                    getString(R.string.server_explanation),
                                    getString(R.string.server_debug_explanation),
                                )
                            } else getString(R.string.server_explanation)
                    }
                )
                addView(input)
            }
        val dialog =
            AlertDialog.Builder(this)
                .setTitle(R.string.server_address)
                .setView(form)
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(R.string.save, null)
                .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val candidate = ServerOrigin.parse(input.text.toString(), BuildConfig.DEBUG)
                if (candidate == null) {
                    input.error = getString(R.string.server_invalid)
                } else if (candidate == origin) {
                    dialog.dismiss()
                } else {
                    AlertDialog.Builder(this)
                        .setTitle(R.string.switch_server_title)
                        .setMessage(getString(R.string.switch_server_detail, candidate.value))
                        .setNegativeButton(R.string.cancel, null)
                        .setPositiveButton(R.string.connect) { _, _ ->
                            dialog.dismiss()
                            clearSession(candidate)
                        }
                        .show()
                }
            }
        }
        dialog.show()
    }

    private fun clearSession(nextOrigin: ServerOrigin? = null) {
        if (auth.replacingSession) return
        if (!BrowserSession.supportsCompleteDeletion) {
            toast(R.string.update_webview)
            return
        }
        auth.cancel()
        cancelFileSelection()
        webView?.clearCache(true)
        destroyWebView()
        DownloadFile.clearExports(cacheDir)
        auth.clearBrowserSession(nextOrigin)
    }

    private fun openExternal(url: String) {
        val uri = runCatching { url.toUri() }.getOrNull()
        if (ServerOrigin.isExternalWebUrl(url)) {
            openBrowser(url)
        } else if (uri?.scheme in listOf("mailto", "tel") && url.none { it.isISOControl() }) {
            try {
                startActivity(
                    Intent(Intent.ACTION_VIEW, uri).addCategory(Intent.CATEGORY_BROWSABLE)
                )
            } catch (_: ActivityNotFoundException) {
                toast(R.string.no_browser)
            }
        } else toast(R.string.link_blocked)
    }

    private fun openBrowser(url: String): Boolean {
        if (!origin.contains(url) && !ServerOrigin.isExternalWebUrl(url)) return false
        return try {
            CustomTabsIntent.Builder().setShowTitle(true).build().launchUrl(this, url.toUri())
            true
        } catch (_: ActivityNotFoundException) {
            toast(R.string.no_browser)
            false
        }
    }

    private fun download(url: String, disposition: String?, mimeType: String?) {
        if (!origin.contains(url)) {
            AlertDialog.Builder(this)
                .setMessage(R.string.download_browser)
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(R.string.open_browser) { _, _ -> openBrowser(lastSafeUrl) }
                .show()
            return
        }
        val generation = documentGeneration
        val name = DownloadFile.safeName(url, disposition, mimeType)
        AlertDialog.Builder(this)
            .setTitle(R.string.download_title)
            .setMessage(getString(R.string.download_detail, name, origin.host))
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(R.string.download) { _, _ ->
                if (generation != documentGeneration || auth.replacingSession)
                    return@setPositiveButton
                toast(R.string.downloading)
                val request =
                    DownloadFile.fetch(this, origin, url, name) { task, result ->
                        httpDownloads.remove(task)
                        if (isDestroyed || generation != documentGeneration) return@fetch
                        result
                            .onSuccess { file ->
                                val uri =
                                    androidx.core.content.FileProvider.getUriForFile(
                                        this,
                                        "$packageName.files",
                                        file,
                                    )
                                val intent =
                                    Intent(Intent.ACTION_SEND)
                                        .setType(mimeType ?: "application/octet-stream")
                                        .putExtra(Intent.EXTRA_STREAM, uri)
                                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                try {
                                    startActivity(
                                        Intent.createChooser(intent, getString(R.string.share_file))
                                    )
                                } catch (_: ActivityNotFoundException) {
                                    toast(R.string.download_failed)
                                }
                            }
                            .onFailure { toast(R.string.download_failed) }
                    }
                httpDownloads.add(request)
            }
            .show()
    }

    private fun toast(message: Int) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()
}
