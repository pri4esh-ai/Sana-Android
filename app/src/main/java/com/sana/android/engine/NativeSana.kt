package com.sana.android.engine

import android.content.Context
import java.io.File

object NativeSana {

    init {
        System.loadLibrary("sana_native")
    }

    external fun nativeInitialize(
        assetManager: android.content.res.AssetManager,
        modelAsset: String,
        cachePath: String,
        preferOpenCl: Boolean,
        cpuThreads: Int
    ): Boolean

    external fun nativeIsInitialized(): Boolean
    external fun nativeGetBackend(): String
    external fun nativeGetStatus(): String
    external fun nativeRelease()

    external fun nativeTestModels(
        transformerPath: String,
        vaePath: String,
        cachePath: String,
        preferOpenCl: Boolean
    ): String

    fun initialize(
        context: Context,
        modelAsset: String,
        preferOpenCl: Boolean = true,
        cpuThreads: Int = 4
    ): Boolean {
        return nativeInitialize(
            context.assets,
            modelAsset,
            context.cacheDir.absolutePath,
            preferOpenCl,
            cpuThreads.coerceIn(1, 8)
        )
    }

    fun isInitialized(): Boolean =
        nativeIsInitialized()

    fun backend(): String =
        nativeGetBackend()

    fun status(): String =
        nativeGetStatus()

    fun release() {
        nativeRelease()
    }

    fun testModels(
        context: Context,
        transformerFile: File,
        vaeFile: File,
        preferOpenCl: Boolean = true
    ): String {
        return nativeTestModels(
            transformerFile.absolutePath,
            vaeFile.absolutePath,
            context.cacheDir.absolutePath,
            preferOpenCl
        )
    }
}
