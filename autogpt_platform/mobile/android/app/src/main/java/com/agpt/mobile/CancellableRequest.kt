package com.agpt.mobile

import java.net.HttpURLConnection
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class CancellableRequest(timeoutMillis: Long, onTimeout: (CancellableRequest) -> Unit = {}) {
    private val outcome = RequestOutcome()
    val cancelled: Boolean
        get() = outcome.cancelled || cancelledByCaller

    @Volatile
    var cancelledByCaller = false
        private set

    private val connection = AtomicReference<HttpURLConnection?>()
    private val future = AtomicReference<Future<*>?>()
    private val deadline =
        deadlines.schedule(
            {
                if (outcome.timeout()) {
                    cancelWork()
                    onTimeout(this)
                }
            },
            timeoutMillis,
            TimeUnit.MILLISECONDS,
        )

    fun attach(value: HttpURLConnection) {
        connection.set(value)
        if (cancelled) {
            connection.compareAndSet(value, null)
            value.disconnect()
            checkActive()
        }
    }

    fun attach(value: Future<*>) {
        if (!outcome.active) {
            if (cancelled) value.cancel(true)
            return
        }
        future.set(value)
        if (!outcome.active) {
            future.compareAndSet(value, null)
            if (cancelled) value.cancel(true)
        }
    }

    fun detach(value: HttpURLConnection) {
        connection.compareAndSet(value, null)
    }

    fun checkActive() {
        if (cancelled) throw CancellationException("Request cancelled")
    }

    fun cancel() {
        cancelledByCaller = true
        deadline.cancel(false)
        if (outcome.cancel()) cancelWork()
    }

    private fun cancelWork() {
        future.getAndSet(null)?.cancel(true)
        connection.getAndSet(null)?.let { value ->
            disconnects.execute { runCatching { value.disconnect() } }
        }
    }

    fun complete(): Boolean {
        if (!outcome.complete()) return false
        deadline.cancel(false)
        connection.set(null)
        future.set(null)
        return true
    }

    companion object {
        private val deadlines =
            ScheduledThreadPoolExecutor(1) { action ->
                    Thread(action, "AutoGPT request deadline").apply { isDaemon = true }
                }
                .apply { removeOnCancelPolicy = true }
        private val disconnects =
            Executors.newFixedThreadPool(2) { action ->
                Thread(action, "AutoGPT request cancellation").apply { isDaemon = true }
            }
    }
}
