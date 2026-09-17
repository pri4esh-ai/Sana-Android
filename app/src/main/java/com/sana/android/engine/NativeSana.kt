package com.sana.android.engine

import android.content.Context
import android.content.res.AssetManager

object NativeSana {

    init {
        System.loadLibrary("sana_native")
    }

    /*
     * ---------------------------------------------------------
     * EXISTING ENGINE API
     * ---------------------------------------------------------
     */

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

    /*
     * ---------------------------------------------------------
     * FD-BASED TRANSFORMER TEST
     * ---------------------------------------------------------
     *
     * fd is owned by native code after the call starts.
     */

    external fun nativeTestTransformerFd(
        transformerFd: Int,
        preferOpenCl: Boolean
    ): String

    /*
     * ---------------------------------------------------------
     * FD-BASED VAE TEST
     * ---------------------------------------------------------
     */

    external fun nativeTestVaeFd(
        vaeFd: Int,
        preferOpenCl: Boolean
    ): String

    /*
     * ---------------------------------------------------------
     * FD-BASED COMBINED TEST
     * ---------------------------------------------------------
     *
     * NO MODEL COPY.
     *
     * Android opens the files.
     * Native receives their Linux file descriptors.
     */

    external fun nativeTestModelsFd(
        transformerFd: Int,
        vaeFd: Int,
        preferOpenCl: Boolean
    ): String

    /*
     * ---------------------------------------------------------
     * KOTLIN HELPERS
     * ---------------------------------------------------------
     */

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

    /*
     * ---------------------------------------------------------
     * TRANSFORMER
     * ---------------------------------------------------------
     */

    fun testTransformer(
        transformerFd: Int,
        preferOpenCl: Boolean = false
    ): String {

        return nativeTestTransformerFd(
            transformerFd,
            preferOpenCl
        )
    }

    /*
     * ---------------------------------------------------------
     * VAE
     * ---------------------------------------------------------
     */

    fun testVae(
        vaeFd: Int,
        preferOpenCl: Boolean = false
    ): String {

        return nativeTestVaeFd(
            vaeFd,
            preferOpenCl
        )
    }

    /*
     * ---------------------------------------------------------
     * TRANSFORMER + VAE
     * ---------------------------------------------------------
     */

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
