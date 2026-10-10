package com.agpt.mobile

import android.webkit.CookieManager
import android.webkit.WebStorage
import androidx.webkit.WebStorageCompat
import androidx.webkit.WebViewFeature

object BrowserSession {
    val supportsCompleteDeletion: Boolean
        get() = WebViewFeature.isFeatureSupported(WebViewFeature.DELETE_BROWSING_DATA)

    fun clear(done: (Boolean) -> Unit) {
        if (WebViewFeature.isFeatureSupported(WebViewFeature.DELETE_BROWSING_DATA)) {
            try {
                WebStorageCompat.deleteBrowsingData(WebStorage.getInstance()) {
                    CookieManager.getInstance().flush()
                    done(true)
                }
            } catch (_: Exception) {
                done(false)
            }
        } else done(false)
    }
}
