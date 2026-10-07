package com.agpt.mobile

import android.content.res.Configuration
import android.os.Bundle
import android.view.ContextThemeWrapper
import androidx.activity.ComponentActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat

class DesignPreviewActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        val configuration =
            Configuration(baseContext.resources.configuration).apply {
                uiMode =
                    (uiMode and Configuration.UI_MODE_NIGHT_MASK.inv()) or
                        if (intent.getBooleanExtra("dark", false)) Configuration.UI_MODE_NIGHT_YES
                        else Configuration.UI_MODE_NIGHT_NO
                fontScale = intent.getFloatExtra("fontScale", 1f).coerceIn(1f, 2f)
            }
        super.onCreate(savedInstanceState)
        val previewContext =
            ContextThemeWrapper(this, R.style.Theme_AutoGPT).apply {
                applyOverrideConfiguration(configuration)
            }
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val layout = BrowserLayout(previewContext)
        layout.back.isEnabled = false
        ViewCompat.setOnApplyWindowInsetsListener(layout) { view, insets ->
            val bars =
                insets.getInsets(
                    WindowInsetsCompat.Type.systemBars() or
                        WindowInsetsCompat.Type.displayCutout() or
                        WindowInsetsCompat.Type.ime()
                )
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        val light = true
        WindowCompat.getInsetsController(window, window.decorView).apply {
            isAppearanceLightStatusBars = light
            isAppearanceLightNavigationBars = light
        }
        setContentView(layout)
        when (intent.getStringExtra("screen")) {
            "loading" -> layout.showLoading()
            "error" ->
                layout.showPanel(
                    R.string.unable_to_load,
                    R.string.load_error_detail,
                    R.string.retry,
                    {},
                    R.string.settings to {},
                )
            else ->
                layout.showPanel(
                    R.string.native_login_title,
                    R.string.sign_in_detail,
                    R.string.continue_browser,
                    {},
                    R.string.open_browser to {},
                )
        }
        if (intent.getStringExtra("screen") == "settings")
            ServerSettingsDialog.create(
                    previewContext,
                    requireNotNull(ServerOrigin.parse(ServerOrigin.DEFAULT, false)),
                ) {}
                .show()
    }
}
