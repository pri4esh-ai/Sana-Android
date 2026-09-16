package com.sana.android.engine

import android.content.Context
import android.content.res.AssetManager
import java.io.File

object NativeSana {

    init {
        System.loadLibrary("sana_native")
    }

    // Existing engine API
    external fun nativeInitialize(
        assetManager: AssetManager,
        modelAsset: String,
        cachePath: String,
        preferOpenCl: Boolean,
        cpuThreads: Int
    ): Boolean

    external fun nativeIsInitialized(): Boolean
    external fun nativeGetBackend(): String
    external fun nativeGetStatus(): String
    external fun nativeRelease()

    // Transformer-only diagnostic
    external fun nativeTestTransformer(
        transformerPath: String,
        cachePath: String,
        preferOpenCl: Boolean
    ): String

    // VAE-only diagnostic
    external fun nativeTestVae(
        vaePath: String,
        cachePath: String,
        preferOpenCl: Boolean
    ): String

    // Legacy combined test (kept for compatibility)
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

    fun isInitialized(): Boolean = nativeIsInitialized()

    fun backend(): String = nativeGetBackend()

    fun status(): String = nativeGetStatus()

    fun release() = nativeRelease()

    fun testTransformer(
        context: Context,
        transformerFile: File,
        preferOpenCl: Boolean = true
    ): String {
        return nativeTestTransformer(
            transformerPath = transformerFile.absolutePath,
            cachePath = context.cacheDir.absolutePath,
            preferOpenCl = preferOpenCl
        )
    }

    fun testVae(
        context: Context,
        vaeFile: File,
        preferOpenCl: Boolean = true
    ): String {
        return nativeTestVae(
            vaePath = vaeFile.absolutePath,
            cachePath = context.cacheDir.absolutePath,
            preferOpenCl = preferOpenCl
        )
    }

    fun testModels(
        context: Context,
        transformerFile: File,
        vaeFile: File,
        preferOpenCl: Boolean = true
    ): String {
        return nativeTestModels(
            transformerPath = transformerFile.absolutePath,
            vaePath = vaeFile.absolutePath,
            cachePath = context.cacheDir.absolutePath,
            preferOpenCl = preferOpenCl
        )
    }
}
