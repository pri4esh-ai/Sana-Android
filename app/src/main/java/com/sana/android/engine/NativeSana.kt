package com.sana.android.engine

import android.content.Context
import android.content.res.AssetManager

object NativeSana {

    init {
        System.loadLibrary("sana_native")
    }

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

    external fun nativeTestTransformerFd(
        transformerFd: Int,
        preferOpenCl: Boolean
    ): String

    external fun nativeTestVaeFd(
        vaeFd: Int,
        preferOpenCl: Boolean
    ): String

    external fun nativeTestModelsFd(
        transformerFd: Int,
        vaeFd: Int,
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

    fun isInitialized(): Boolean {
        return nativeIsInitialized()
    }

    fun backend(): String {
        return nativeGetBackend()
    }

    fun status(): String {
        return nativeGetStatus()
    }

    fun release() {
        nativeRelease()
    }

    fun testTransformer(
        transformerFd: Int,
        preferOpenCl: Boolean = false
    ): String {

        return nativeTestTransformerFd(
            transformerFd,
            preferOpenCl
        )
    }

    fun testVae(
        vaeFd: Int,
        preferOpenCl: Boolean = false
    ): String {

        return nativeTestVaeFd(
            vaeFd,
            preferOpenCl
        )
    }

    fun testModels(
        transformerFd: Int,
        vaeFd: Int,
        preferOpenCl: Boolean = false
    ): String {

        return nativeTestModelsFd(
            transformerFd,
            vaeFd,
            preferOpenCl
        )
    }
}
