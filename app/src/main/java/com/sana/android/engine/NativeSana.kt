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

    /**
     * Transformer-only diagnostic.
     *
     * IMPORTANT:
     * - Loads ONLY sana_transformer.mnn
     * - Does NOT load the VAE
     * - Does NOT create a second Transformer session
     */
    external fun nativeTestTransformer(
        transformerPath: String,
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
}
