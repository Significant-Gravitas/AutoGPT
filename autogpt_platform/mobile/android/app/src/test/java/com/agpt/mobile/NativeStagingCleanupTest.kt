package com.agpt.mobile

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class NativeStagingCleanupTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test
    fun removesOnlyTheAppStagingFileFamily() {
        val cache = temporary.newFolder("cache")
        val root = File(cache, "native-downloads").apply { mkdir() }
        val stale = File(root, "export-old.partial").apply { writeText("interrupted export") }
        val unrelated = File(root, "keep.txt").apply { writeText("preserve") }
        val outside = File(cache, "export-outside.partial").apply { writeText("preserve") }
        val directory = File(root, "export-directory.partial").apply { mkdir() }
        val nested = File(directory, "export-nested.partial").apply { writeText("preserve") }
        NativeStagingCleanup().runOnce(cache)
        assertFalse(stale.exists())
        assertTrue(unrelated.exists())
        assertTrue(outside.exists())
        assertTrue(nested.exists())
    }

    @Test
    fun laterInvocationsPreserveCurrentProcessTransfers() {
        val cache = temporary.newFolder("cache")
        val root = File(cache, "native-downloads").apply { mkdir() }
        val cleanup = NativeStagingCleanup()
        cleanup.runOnce(cache)
        val active = File(root, "export-active.partial").apply { writeText("active transfer") }
        cleanup.runOnce(cache)
        assertTrue(active.exists())
        NativeStagingCleanup().runOnce(cache)
        assertFalse(active.exists())
    }
}
