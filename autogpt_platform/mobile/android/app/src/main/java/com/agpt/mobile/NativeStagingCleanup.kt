package com.agpt.mobile

import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

class NativeStagingCleanup {
    private val started = AtomicBoolean(false)

    fun runOnce(cache: File) {
        if (!started.compareAndSet(false, true)) return
        File(cache, "native-downloads").listFiles()?.forEach { file ->
            if (file.isFile && file.name.startsWith("export-") && file.name.endsWith(".partial")) {
                file.delete()
            }
        }
    }
}
