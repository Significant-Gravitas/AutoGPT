package com.agpt.mobile

import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Test

class CancellableRequestTest {
    @Test
    fun cancelDisconnectsTheOwnedConnection() {
        val disconnected = CountDownLatch(1)
        val request = CancellableRequest(30_000)
        request.attach(connection(disconnected))
        request.cancel()
        assertTrue(disconnected.await(2, TimeUnit.SECONDS))
        assertTrue(request.cancelledByCaller)
        assertThrows(CancellationException::class.java) { request.checkActive() }
    }

    @Test
    fun canceledRequestsCannotStartAnotherConnection() {
        val disconnected = CountDownLatch(1)
        val request = CancellableRequest(30_000)
        request.cancel()
        assertThrows(CancellationException::class.java) { request.attach(connection(disconnected)) }
        assertTrue(disconnected.await(2, TimeUnit.SECONDS))
    }

    @Test
    fun absoluteDeadlineCancelsEvenWithoutFurtherReadActivity() {
        val expired = CountDownLatch(1)
        val request = CancellableRequest(20) { expired.countDown() }
        assertTrue(expired.await(2, TimeUnit.SECONDS))
        assertTrue(request.cancelled)
        assertFalse(request.cancelledByCaller)
    }

    private fun connection(disconnected: CountDownLatch) =
        object : HttpURLConnection(URL("https://example.test")) {
            override fun connect() = Unit

            override fun disconnect() {
                disconnected.countDown()
            }

            override fun usingProxy() = false
        }
}
