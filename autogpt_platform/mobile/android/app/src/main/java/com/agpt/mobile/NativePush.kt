package com.agpt.mobile

import android.Manifest
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.webkit.WebView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.webkit.JavaScriptReplyProxy
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import com.google.firebase.FirebaseApp
import com.google.firebase.messaging.FirebaseMessaging
import java.util.UUID
import org.json.JSONObject

class NativePush(private val activity: ComponentActivity) {
    private val preferences by lazy { activity.getSharedPreferences("push", Context.MODE_PRIVATE) }
    private var view: WebView? = null
    private var generation = 0
    private var pendingPermission: ((Boolean) -> Unit)? = null
    private val permission =
        activity.registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            val callback = pendingPermission
            pendingPermission = null
            callback?.invoke(granted)
        }

    fun attach(browser: WebView, origin: ServerOrigin) {
        view = browser
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) return
        WebViewCompat.addWebMessageListener(browser, "AutoGPTPush", setOf(origin.value)) {
            source,
            message,
            sourceOrigin,
            mainFrame,
            reply ->
            if (
                source !== view ||
                    !mainFrame ||
                    !origin.contains(sourceOrigin.toString()) ||
                    !origin.contains(source.url ?: "")
            )
                return@addWebMessageListener
            val raw = message.data ?: return@addWebMessageListener
            if (raw.length > 2048) return@addWebMessageListener
            val request =
                runCatching { JSONObject(raw) }.getOrNull() ?: return@addWebMessageListener
            handle(request, origin, reply)
        }
    }

    fun navigation() {
        generation++
    }

    fun detach() {
        generation++
        view = null
        pendingPermission = null
    }

    fun clear() {
        generation++
        if (FirebaseApp.getApps(activity).isNotEmpty())
            FirebaseMessaging.getInstance().isAutoInitEnabled = false
        PushRegistration.remove(activity)
        preferences.edit().remove("binding").putBoolean("enabled", false).apply()
        activity.getSystemService(NotificationManager::class.java).cancelAll()
    }

    private fun handle(request: JSONObject, origin: ServerOrigin, reply: JavaScriptReplyProxy) {
        val id = request.optString("id")
        if (runCatching { UUID.fromString(id) }.isFailure) return
        val account = request.optString("account_id")
        val action = request.optString("action")
        if (account.length > 128 || action !in setOf("status", "enable", "disable")) return
        if (
            preferences.getString("account", "") != account ||
                preferences.getString("origin", "") != origin.value
        ) {
            clear()
            preferences
                .edit()
                .putString("account", account)
                .putString("origin", origin.value)
                .apply()
        }
        if (action == "disable" || account.isEmpty()) {
            clear()
            respond(reply, id, "disabled")
            return
        }
        if (action == "status" && !preferences.getBoolean("enabled", false)) {
            respond(reply, id, "disabled")
            return
        }
        if (FirebaseApp.getApps(activity).isEmpty()) {
            respond(reply, id, "unavailable")
            return
        }
        val current = generation
        val proceed: (Boolean) -> Unit = { granted ->
            if (current == generation && view != null) {
                if (granted) token(reply, id, current) else respond(reply, id, "denied")
            }
        }
        if (
            action == "enable" &&
                Build.VERSION.SDK_INT >= 33 &&
                ContextCompat.checkSelfPermission(
                    activity,
                    Manifest.permission.POST_NOTIFICATIONS,
                ) != PackageManager.PERMISSION_GRANTED
        ) {
            if (pendingPermission != null) {
                respond(reply, id, "unavailable")
                return
            }
            pendingPermission = proceed
            permission.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else proceed(NotificationManagerCompat.from(activity).areNotificationsEnabled())
    }

    private fun token(reply: JavaScriptReplyProxy, id: String, current: Int) {
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (current != generation || view == null) return@addOnCompleteListener
            if (!task.isSuccessful) {
                respond(reply, id, "unavailable")
                return@addOnCompleteListener
            }
            val binding = preferences.getString("binding", null) ?: UUID.randomUUID().toString()
            preferences.edit().putString("binding", binding).putBoolean("enabled", true).apply()
            FirebaseMessaging.getInstance().isAutoInitEnabled = true
            reply.postMessage(
                JSONObject()
                    .put("id", id)
                    .put("permission", "granted")
                    .put("provider", "fcm")
                    .put("environment", "production")
                    .put("token", task.result)
                    .put("binding_id", binding)
                    .toString()
            )
        }
    }

    private fun respond(reply: JavaScriptReplyProxy, id: String, status: String) {
        reply.postMessage(JSONObject().put("id", id).put("permission", status).toString())
    }
}
