package com.agpt.mobile

import java.util.concurrent.atomic.AtomicReference

internal class RequestOutcome {
    private enum class State {
        ACTIVE,
        COMPLETED,
        TIMED_OUT,
        CANCELLED,
    }

    private val state = AtomicReference(State.ACTIVE)
    val active: Boolean
        get() = state.get() == State.ACTIVE

    val cancelled: Boolean
        get() = state.get() in setOf(State.TIMED_OUT, State.CANCELLED)

    fun complete(): Boolean = state.compareAndSet(State.ACTIVE, State.COMPLETED)

    fun timeout(): Boolean = state.compareAndSet(State.ACTIVE, State.TIMED_OUT)

    fun cancel(): Boolean = state.compareAndSet(State.ACTIVE, State.CANCELLED)
}
