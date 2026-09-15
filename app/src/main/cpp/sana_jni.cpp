#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsAvailable(
        JNIEnv* env,
        jobject thiz
) {
    LOGI("Sana native library loaded");

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject thiz
) {
    const char* backend =
            "MNN / OpenCL / ARM FP16";

    return env->NewStringUTF(backend);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeMnnAvailable(
        JNIEnv* env,
        jobject thiz
) {
    try {
        MNN::Interpreter* interpreter =
                MNN::Interpreter::createFromBuffer(
                        nullptr,
                        0
                );

        if (interpreter != nullptr) {
            delete interpreter;

            LOGI("MNN runtime is linked successfully");

            return JNI_TRUE;
        }
    } catch (...) {
        LOGE("MNN runtime test failed");
    }

    return JNI_FALSE;
}
