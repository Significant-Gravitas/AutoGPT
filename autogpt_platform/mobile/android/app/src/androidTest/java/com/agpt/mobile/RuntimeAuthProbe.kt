package com.agpt.mobile

import android.app.Application
import android.content.Context
import android.net.Uri
import android.webkit.CookieManager
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import org.json.JSONObject

class RuntimeAuthProbe(private val context: Context, private val runtime: RuntimeSupport) {
    fun run(value: String) {
        val origin = requireNotNull(ServerOrigin.parse(value, true))
        check(origin.value == value)
        check(Uri.parse(value).host in setOf("localhost", "127.0.0.1", "10.0.2.2")) {
            "Runtime authentication only supports the disposable local fixture"
        }
        check(request(origin, "/health").optBoolean("fixture")) {
            "The target is not the labelled native integration fixture"
        }
        val store = ViewModelStore()
        val model = runtime.main {
            ViewModelProvider(
                store,
                ViewModelProvider.AndroidViewModelFactory(
                    context.applicationContext as Application
                ),
            )[AuthViewModel::class.java]
        }
        var cookies: String? = null
        try {
            val start = Uri.parse(runtime.main { model.start(origin) })
            val callback =
                request(
                        origin,
                        "/api/auth/mobile/authorize",
                        JSONObject()
                            .put("code_challenge", start.getQueryParameter("code_challenge"))
                            .put("state", start.getQueryParameter("state"))
                            .put("expected_user_id", "fixture-user"),
                    )
                    .getString("url")
            check(
                runtime.callback<Boolean> { completed ->
                    CookieManager.getInstance().setCookie(
                        origin.value,
                        "runtime_old_account=disposable; Path=/; HttpOnly; Max-Age=120",
                    ) {
                        completed.complete(it)
                    }
                }
            )
            val installed = CompletableFuture<Unit>()
            runtime.main {
                model.listener = {
                    when (model.status) {
                        AuthViewModel.Status.READY_TO_INSTALL -> model.installReadySession()
                        AuthViewModel.Status.SUCCESS -> installed.complete(Unit)
                        AuthViewModel.Status.FAILED,
                        AuthViewModel.Status.EXPIRED ->
                            installed.completeExceptionally(
                                AssertionError("Native fixture exchange failed")
                            )
                        else -> Unit
                    }
                }
                model.acceptCallback(callback, origin)
            }
            installed.get(35, TimeUnit.SECONDS)
            cookies = runtime.main { CookieManager.getInstance().getCookie(origin.value) }
            check(!cookies.orEmpty().contains("runtime_old_account=")) {
                "Old-account cookie survived successful installation"
            }
            val session = request(origin, "/api/fixture/session", cookie = cookies)
            check(session.getBoolean("authenticated"))
            check(session.getBoolean("cacheCookieReceived"))
            runtime.main { model.acknowledge() }
            val cancelledStart = Uri.parse(runtime.main { model.start(origin) })
            runtime.main {
                model.acceptCallback(
                    "autogpt://auth/callback?error=access_denied&state=${cancelledStart.getQueryParameter("state")}",
                    origin,
                )
            }
            check(runtime.main { model.status } == AuthViewModel.Status.CANCELED)
            check(runtime.main { CookieManager.getInstance().getCookie(origin.value) } == cookies) {
                "Cancelled sign-in replaced the prior session"
            }
        } finally {
            runtime.main { store.clear() }
            cookies?.let { request(origin, "/api/fixture/logout", JSONObject(), it) }
            check(
                runtime.callback<Boolean> { completed ->
                    BrowserSession.clear { completed.complete(it) }
                }
            )
        }
    }

    private fun request(
        origin: ServerOrigin,
        path: String,
        body: JSONObject? = null,
        cookie: String? = null,
    ): JSONObject {
        val connection = URL(origin.value + path).openConnection() as HttpURLConnection
        try {
            connection.instanceFollowRedirects = false
            connection.connectTimeout = 5_000
            connection.readTimeout = 5_000
            connection.setRequestProperty("Origin", origin.value)
            cookie?.let { connection.setRequestProperty("Cookie", it) }
            if (body != null) {
                connection.requestMethod = "POST"
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json")
                val data = body.toString().toByteArray()
                connection.setFixedLengthStreamingMode(data.size)
                connection.outputStream.use { it.write(data) }
            }
            check(connection.responseCode == 200) {
                "Local fixture returned HTTP ${connection.responseCode}"
            }
            val bytes = ByteArray(16 * 1024)
            val length =
                connection.inputStream.use { input ->
                    var total = 0
                    while (total < bytes.size) {
                        val count = input.read(bytes, total, bytes.size - total)
                        if (count < 0) break
                        total += count
                    }
                    total
                }
            check(length < bytes.size) { "Unexpectedly large fixture response" }
            return JSONObject(String(bytes, 0, length, Charsets.UTF_8))
        } finally {
            connection.disconnect()
        }
    }
}
