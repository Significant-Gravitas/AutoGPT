package com.agpt.mobile

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.webkit.CookieManager
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import org.json.JSONObject

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    enum class Status {
        IDLE,
        WAITING,
        EXCHANGING,
        READY_TO_INSTALL,
        INSTALLING,
        CLEARING_SESSION,
        SESSION_CLEARED,
        SUCCESS,
        FAILED,
        EXPIRED,
        CANCELED,
    }

    var status = Status.IDLE
        private set

    var listener: (() -> Unit)? = null
    private var pending: AuthSession? = null
    private var generation = 0

    private data class Installation(
        val cookies: List<String>,
        val origin: ServerOrigin,
        val generation: Int,
    )

    private var installation: Installation? = null
    val replacingSession: Boolean
        get() =
            status in
                setOf(
                    Status.READY_TO_INSTALL,
                    Status.INSTALLING,
                    Status.CLEARING_SESSION,
                    Status.SESSION_CLEARED,
                    Status.SUCCESS,
                )

    private val main = Handler(Looper.getMainLooper())
    private val executor = Executors.newFixedThreadPool(2)
    private var activeRequest: CancellableRequest? = null

    fun start(origin: ServerOrigin): String {
        check(!replacingSession)
        activeRequest?.cancel()
        activeRequest = null
        installation = null
        generation++
        val session = AuthSession.create(origin, SystemClock.elapsedRealtime())
        pending = session
        update(Status.WAITING)
        return session.startUrl
    }

    fun acceptCallback(callback: String, origin: ServerOrigin) {
        if (replacingSession || status == Status.EXCHANGING) return
        val session = pending
        if (session == null || session.origin != origin) {
            update(Status.EXPIRED)
            return
        }
        val result = session.callback(callback, SystemClock.elapsedRealtime())
        if (result == null) {
            update(Status.EXPIRED)
            return
        }
        pending = null
        if (result == AuthSession.Callback.Canceled) {
            generation++
            update(Status.CANCELED)
            return
        }
        val code = (result as AuthSession.Callback.Code).value
        val requestGeneration = ++generation
        update(Status.EXCHANGING)
        val request =
            CancellableRequest(30_000) { expired ->
                main.post {
                    if (
                        generation == requestGeneration &&
                            activeRequest === expired &&
                            status == Status.EXCHANGING
                    ) {
                        generation++
                        activeRequest = null
                        update(Status.FAILED)
                    }
                }
            }
        activeRequest = request
        val future = executor.submit {
            val cookies = runCatching { exchange(session, code, request) }.getOrNull()
            if (!request.complete()) return@submit
            main.post {
                if (
                    generation != requestGeneration ||
                        activeRequest !== request ||
                        request.cancelledByCaller ||
                        status != Status.EXCHANGING
                )
                    return@post
                activeRequest = null
                if (cookies.isNullOrEmpty()) {
                    update(Status.FAILED)
                    return@post
                }
                installation = Installation(cookies, origin, requestGeneration)
                update(Status.READY_TO_INSTALL)
            }
        }
        request.attach(future)
    }

    fun installReadySession() {
        val next = installation ?: return
        if (next.generation != generation || status != Status.READY_TO_INSTALL) return
        installation = null
        update(Status.INSTALLING)
        BrowserSession.clear { succeeded ->
            if (next.generation != generation) return@clear
            if (succeeded) installCookies(next.cookies, next.origin, next.generation)
            else update(Status.FAILED)
        }
    }

    fun clearBrowserSession(nextOrigin: ServerOrigin?) {
        if (replacingSession) return
        cancel()
        val requestGeneration = ++generation
        update(Status.CLEARING_SESSION)
        BrowserSession.clear { succeeded ->
            if (generation != requestGeneration) return@clear
            if (succeeded) {
                nextOrigin?.let { server ->
                    getApplication<Application>().getSharedPreferences("server", 0).edit {
                        putString("origin", server.value)
                    }
                }
                update(Status.SESSION_CLEARED)
            } else update(Status.FAILED)
        }
    }

    fun cancel() {
        if (replacingSession) return
        activeRequest?.cancel()
        activeRequest = null
        installation = null
        generation++
        pending = null
        update(Status.IDLE)
    }

    fun acknowledge() {
        if (
            status in
                listOf(
                    Status.SUCCESS,
                    Status.FAILED,
                    Status.EXPIRED,
                    Status.CANCELED,
                    Status.SESSION_CLEARED,
                )
        )
            update(Status.IDLE)
    }

    private fun exchange(
        session: AuthSession,
        code: String,
        request: CancellableRequest,
    ): List<String> {
        val connection =
            URL("${session.origin.value}/api/auth/mobile/exchange").openConnection()
                as HttpURLConnection
        request.attach(connection)
        try {
            request.checkActive()
            connection.instanceFollowRedirects = false
            connection.connectTimeout = 20_000
            connection.readTimeout = 20_000
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Accept", "application/json")
            connection.setRequestProperty("Origin", session.origin.value)
            val body =
                JSONObject()
                    .put("code", code)
                    .put("code_verifier", session.verifier)
                    .toString()
                    .toByteArray()
            connection.setFixedLengthStreamingMode(body.size)
            connection.outputStream.use { it.write(body) }
            check(connection.responseCode == HttpURLConnection.HTTP_OK)
            request.checkActive()
            val headers =
                connection.headerFields.entries
                    .filter { it.key.equals("Set-Cookie", true) }
                    .flatMap { it.value }
            check(headers.isNotEmpty())
            val cookies = headers.map {
                requireNotNull(SessionCookie.forOrigin(it, session.origin))
            }
            check(SessionCookie.hasUsableSessionToken(cookies, System.currentTimeMillis()))
            return cookies
        } finally {
            request.detach(connection)
            connection.disconnect()
        }
    }

    private fun installCookies(
        cookies: List<String>,
        origin: ServerOrigin,
        requestGeneration: Int,
    ) {
        val manager = CookieManager.getInstance()
        var remaining = cookies.size
        var succeeded = true
        for (cookie in cookies) {
            manager.setCookie(origin.value, cookie) { accepted ->
                if (generation != requestGeneration) return@setCookie
                succeeded = succeeded && accepted
                remaining--
                if (remaining == 0) {
                    manager.flush()
                    if (succeeded) update(Status.SUCCESS)
                    else
                        BrowserSession.clear {
                            if (generation == requestGeneration) update(Status.FAILED)
                        }
                }
            }
        }
    }

    private fun update(value: Status) {
        status = value
        listener?.invoke()
    }

    override fun onCleared() {
        generation++
        activeRequest?.cancel()
        activeRequest = null
        executor.shutdownNow()
        listener = null
    }
}
