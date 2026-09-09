package com.agpt.mobile

import android.app.Activity
import android.app.Instrumentation
import android.os.Bundle
import android.webkit.CookieManager
import android.webkit.WebView
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature

class RuntimeProbe : Instrumentation() {
    private var fixtureOrigin: String? = null
    private val result = Bundle()
    private var passed = 0

    override fun onCreate(arguments: Bundle?) {
        super.onCreate(arguments)
        if (arguments?.getString("disposable") != "true") {
            result.putString("probe_status", "REFUSED")
            result.putString(
                "stream",
                "This probe clears app browsing data. Run only on a disposable test device with -e disposable true.",
            )
            finish(Activity.RESULT_CANCELED, result)
            return
        }
        fixtureOrigin = arguments.getString("fixtureOrigin")
        start()
    }

    override fun onStart() {
        val runtime = RuntimeSupport(this)
        var view: WebView? = null
        var success = false
        try {
            view = runtime.main { WebView(targetContext) }
            result.putString(
                "webview_version",
                WebViewCompat.getCurrentWebViewPackage(targetContext)?.versionName,
            )
            verify("webview_capabilities") {
                runtime.main {
                    check(WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER))
                    check(BrowserSession.supportsCompleteDeletion)
                }
            }
            verify("cookie_deletion_callback") {
                val origin = "https://android-runtime.invalid/"
                check(
                    runtime.callback<Boolean> { completed ->
                        CookieManager.getInstance().setCookie(
                            origin,
                            "runtime_probe=disposable; Path=/; Secure; HttpOnly; Max-Age=120",
                        ) {
                            completed.complete(it)
                        }
                    }
                )
                check(
                    runtime.main {
                        CookieManager.getInstance()
                            .getCookie(origin)
                            .orEmpty()
                            .contains("runtime_probe=")
                    }
                )
                check(
                    runtime.callback<Boolean> { completed ->
                        BrowserSession.clear { completed.complete(it) }
                    }
                )
                check(
                    runtime.main {
                        !CookieManager.getInstance()
                            .getCookie(origin)
                            .orEmpty()
                            .contains("runtime_probe=")
                    }
                )
            }
            runtime.main { view?.destroy() }
            view = null
            fixtureOrigin?.let { value ->
                verify("fixture_auth_exchange_and_cancellation") {
                    RuntimeAuthProbe(targetContext, runtime).run(value)
                }
            }
            verify("native_download_origin_boundary") {
                RuntimeDownloadProbe(targetContext, runtime).run()
            }
            result.putString("probe_status", "PASS")
            result.putString("suite", if (fixtureOrigin == null) "platform" else "fixture")
            result.putString(
                "stream",
                "\nRuntime probe PASS: $passed checks; WebView ${result.getString("webview_version")}\n",
            )
            success = true
        } catch (error: Throwable) {
            result.putString("probe_status", "FAIL")
            result.putString(
                "stream",
                "\nRuntime probe FAIL after $passed checks: ${error.javaClass.simpleName}: ${error.message}\n",
            )
        } finally {
            runtime.main { view?.destroy() }
            result.putInt("checks_passed", passed)
            finish(if (success) Activity.RESULT_OK else Activity.RESULT_CANCELED, result)
        }
    }

    private fun verify(name: String, action: () -> Unit) {
        result.putString("current_check", name)
        action()
        result.putString(name, "PASS")
        passed++
        sendStatus(0, Bundle().apply { putString("stream", "$name: PASS\n") })
    }
}
