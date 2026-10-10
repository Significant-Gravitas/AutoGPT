package com.agpt.mobile

import android.os.Bundle
import android.webkit.WebView
import androidx.activity.ComponentActivity

class PushProbeActivity : ComponentActivity() {
    lateinit var browser: WebView
        private set

    private val push = NativePush(this)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        browser = WebView(this)
        browser.settings.javaScriptEnabled = true
        setContentView(browser)
        push.attach(
            browser,
            requireNotNull(ServerOrigin.parse("https://push-runtime.invalid", false)),
        )
    }

    override fun onDestroy() {
        push.detach()
        browser.destroy()
        super.onDestroy()
    }
}
