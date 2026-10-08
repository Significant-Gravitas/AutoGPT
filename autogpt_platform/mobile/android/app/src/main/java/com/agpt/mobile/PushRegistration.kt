package com.agpt.mobile

import android.content.Context
import android.webkit.CookieManager
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import org.json.JSONObject

object PushRegistration {
    private val network = Executors.newSingleThreadExecutor()

    fun sync(context: Context, token: String) {
        val prefs = context.getSharedPreferences("push", Context.MODE_PRIVATE)
        if (!prefs.getBoolean("enabled", false)) return
        send(
            context,
            "",
            JSONObject()
                .put("provider", "fcm")
                .put("environment", "production")
                .put("token", token)
                .put("binding_id", prefs.getString("binding", ""))
                .put("expected_user_id", prefs.getString("account", "")),
        )
    }

    fun remove(context: Context) {
        val prefs = context.getSharedPreferences("push", Context.MODE_PRIVATE)
        val binding = prefs.getString("binding", "") ?: ""
        if (binding.isNotEmpty()) send(context, "/remove", JSONObject().put("binding_id", binding))
    }

    private fun send(context: Context, suffix: String, payload: JSONObject) {
        val origin =
            ServerOrigin.parse(
                context.getSharedPreferences("push", Context.MODE_PRIVATE).getString("origin", "")
                    ?: "",
                BuildConfig.DEBUG,
            ) ?: return
        val cookie = CookieManager.getInstance().getCookie(origin.value) ?: return
        network.execute {
            runCatching {
                val connection =
                    URL("${origin.value}/api/auth/mobile/push$suffix").openConnection()
                        as HttpURLConnection
                try {
                    connection.requestMethod = "POST"
                    connection.instanceFollowRedirects = false
                    connection.connectTimeout = 15_000
                    connection.readTimeout = 15_000
                    connection.doOutput = true
                    connection.setRequestProperty("Origin", origin.value)
                    connection.setRequestProperty("Cookie", cookie)
                    connection.setRequestProperty("Content-Type", "application/json")
                    connection.outputStream.use { it.write(payload.toString().toByteArray()) }
                    connection.responseCode
                } finally {
                    connection.disconnect()
                }
            }
        }
    }
}
