package com.sana.android.engine

object NativeSana {

    init {
        System.loadLibrary("sana_native")
    }

    external fun nativeIsAvailable(): Boolean

    external fun nativeGetBackend(): String

    fun isAvailable(): Boolean = nativeIsAvailable()

    fun backend(): String = nativeGetBackend()
}
