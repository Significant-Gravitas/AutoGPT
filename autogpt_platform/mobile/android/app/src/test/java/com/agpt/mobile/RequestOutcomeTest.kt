package com.agpt.mobile

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Test

class RequestOutcomeTest {
    @Test
    fun completionPreventsLateTimeoutAndCancellation() {
        val outcome = RequestOutcome()
        assertTrue(outcome.complete())
        assertFalse(outcome.timeout())
        assertFalse(outcome.cancel())
        assertFalse(outcome.cancelled)
    }

    @Test
    fun timeoutPreventsLateSuccess() {
        val outcome = RequestOutcome()
        assertTrue(outcome.timeout())
        assertFalse(outcome.complete())
        assertTrue(outcome.cancelled)
    }

    @Test
    fun cancellationPreventsBothOtherTerminalOutcomes() {
        val outcome = RequestOutcome()
        assertTrue(outcome.cancel())
        assertFalse(outcome.complete())
        assertFalse(outcome.timeout())
    }

    @Test
    fun concurrentCompletionAndTimeoutHaveExactlyOneWinner() {
        val workers = Executors.newFixedThreadPool(2)
        try {
            repeat(100) {
                val outcome = RequestOutcome()
                val gate = CountDownLatch(1)
                val success =
                    workers.submit<Boolean> {
                        gate.await()
                        outcome.complete()
                    }
                val timeout =
                    workers.submit<Boolean> {
                        gate.await()
                        outcome.timeout()
                    }
                gate.countDown()
                assertTrue(success.get(2, TimeUnit.SECONDS) xor timeout.get(2, TimeUnit.SECONDS))
            }
        } finally {
            workers.shutdownNow()
        }
    }
}
