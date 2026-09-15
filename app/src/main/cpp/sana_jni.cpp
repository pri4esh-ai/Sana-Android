#include <jni.h>
#include <android/log.h>

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
    return env->NewStringUTF(
            "MNN / OpenCL / ARM FP16"
    );
}
