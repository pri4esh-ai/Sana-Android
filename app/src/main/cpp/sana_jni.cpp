#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

/**
 * Persistent MNN engine.
 *
 * The interpreter and session remain alive between generations.
 *
 * This avoids repeatedly doing:
 *
 *   load model
 *   create interpreter
 *   create session
 *   inference
 *   destroy
 *
 * Instead:
 *
 *   initialize once
 *       |
 *       +--> interpreter
 *       |
 *       +--> session
 *
 *   Generate 1
 *   Generate 2
 *   Generate 3
 *   ...
 */
class SanaEngine {

public:

    SanaEngine() = default;

    ~SanaEngine() {
        release();
    }

    bool initialize(
            AAssetManager* assetManager,
            const std::string& assetName,
            const std::string& cachePath,
            bool preferOpenCl,
            int cpuThreads
    ) {

        std::lock_guard<std::mutex> lock(mMutex);

        if (mInitialized) {

            LOGI(
                    "Sana engine already initialized."
            );

            return true;
        }

        if (assetManager == nullptr) {

            setError(
                    "AssetManager is null."
            );

            LOGE(
                    "AssetManager is null."
            );

            return false;
        }

        if (assetName.empty()) {

            setError(
                    "Model asset path is empty."
            );

            LOGE(
                    "Model asset path is empty."
            );

            return false;
        }

        LOGI(
                "========================================"
        );

        LOGI(
                "SANA MNN ENGINE INITIALIZATION"
        );

        LOGI(
                "========================================"
        );

        LOGI(
                "Model asset: %s",
                assetName.c_str()
        );

        /*
         * Open model from Android assets.
         */
        AAsset* asset =
                AAssetManager_open(
                        assetManager,
                        assetName.c_str(),
                        AASSET_MODE_BUFFER
                );

        if (asset == nullptr) {

            setError(
                    "Unable to open model asset: " +
                    assetName
            );

            LOGE(
                    "Could not open model asset: %s",
                    assetName.c_str()
            );

            return false;
        }

        const off_t assetLength =
                AAsset_getLength(asset);

        if (assetLength <= 0) {

            AAsset_close(asset);

            setError(
                    "Model asset is empty."
            );

            LOGE(
                    "Model asset is empty."
            );

            return false;
        }

        LOGI(
                "Model size: %.2f MB",
                static_cast<double>(assetLength) /
                (1024.0 * 1024.0)
        );

        /*
         * Keep model memory alive for the complete
         * lifetime of the MNN interpreter.
         */
        mModelBuffer.resize(
                static_cast<size_t>(assetLength)
        );

        const int64_t bytesRead =
                AAsset_read(
                        asset,
                        mModelBuffer.data(),
                        static_cast<size_t>(assetLength)
                );

        AAsset_close(asset);

        if (bytesRead !=
            static_cast<int64_t>(assetLength)) {

            mModelBuffer.clear();

            setError(
                    "Failed to read complete MNN model."
            );

            LOGE(
                    "Model read failed. Expected=%lld Actual=%lld",
                    static_cast<long long>(assetLength),
                    static_cast<long long>(bytesRead)
            );

            return false;
        }

        LOGI(
                "Model loaded into native memory."
        );

        /*
         * Create MNN interpreter.
         */
        mInterpreter.reset(
                MNN::Interpreter::createFromBuffer(
                        mModelBuffer.data(),
                        mModelBuffer.size()
                )
        );

        if (!mInterpreter) {

            mModelBuffer.clear();

            setError(
                    "MNN Interpreter creation failed."
            );

            LOGE(
                    "MNN Interpreter::createFromBuffer failed."
            );

            return false;
        }

        LOGI(
                "MNN interpreter created."
        );

        /*
         * Configure persistent MNN cache.
         *
         * Useful for OpenCL kernel/tuning reuse.
         */
        if (!cachePath.empty()) {

            mCacheFile =
                    cachePath +
                    "/sana_mnn_gpu.cache";

            mInterpreter->setCacheFile(
                    mCacheFile.c_str()
            );

            LOGI(
                    "MNN cache: %s",
                    mCacheFile.c_str()
            );
        }

        /*
         * First choice:
         *
         * ARM GPU / OpenCL.
         */
        if (preferOpenCl) {

            LOGI(
                    "Trying OpenCL backend..."
            );

            if (createSession(
                    MNN_FORWARD_OPENCL,
                    1,
                    true
            )) {

                mBackend =
                        "OpenCL / Low Precision";

                mInitialized =
                        true;

                LOGI(
                        "OpenCL session successfully created."
                );

                LOGI(
                        "Sana backend: OpenCL / Low Precision"
                );

                return true;
            }

            LOGI(
                    "OpenCL session unavailable."
            );

            LOGI(
                    "Falling back to ARM CPU."
            );

            destroySession();
        }

        /*
         * CPU fallback.
         */
        const int safeThreads =
                std::max(
                        1,
                        std::min(
                                cpuThreads,
                                8
                        )
                );

        LOGI(
                "Trying ARM CPU backend..."
        );

        if (!createSession(
                MNN_FORWARD_CPU,
                safeThreads,
                true
        )) {

            destroySession();

            mInterpreter.reset();

            mModelBuffer.clear();

            setError(
                    "Unable to create OpenCL or CPU MNN session."
            );

            LOGE(
                    "No usable MNN backend could be created."
            );

            return false;
        }

        mBackend =
                "ARM CPU / Low Precision";

        mInitialized =
                true;

        LOGI(
                "ARM CPU session successfully created."
        );

        LOGI(
                "Sana backend: ARM CPU / Low Precision"
        );

        LOGI(
                "CPU threads: %d",
                safeThreads
        );

        return true;
    }


    bool createSession(
            MNNForwardType backend,
            int threads,
            bool lowPrecision
    ) {

        if (!mInterpreter) {

            LOGE(
                    "Cannot create session: interpreter is null."
            );

            return false;
        }

        MNN::ScheduleConfig config;

        config.type =
                backend;

        /*
         * CPU:
         *     number of CPU threads.
         *
         * OpenCL:
         *     keep conservative because this field
         *     has different semantics on GPU.
         */
        if (backend ==
            MNN_FORWARD_CPU) {

            config.numThread =
                    threads;

        } else {

            config.numThread =
                    1;
        }

        /*
         * Backend precision configuration.
         *
         * Precision_Low allows MNN to use lower precision
         * execution where supported.
         */
        MNN::BackendConfig backendConfig;

        if (lowPrecision) {

            backendConfig.precision =
                    MNN::BackendConfig::Precision_Low;

            config.backendConfig =
                    &backendConfig;
        }

        LOGI(
                "Creating MNN session. backend=%d",
                static_cast<int>(backend)
        );

        MNN::Session* session =
                mInterpreter->createSession(
                        config
                );

        if (session == nullptr) {

            LOGE(
                    "MNN createSession failed. backend=%d",
                    static_cast<int>(backend)
            );

            return false;
        }

        mSession =
                session;

        LOGI(
                "MNN session created."
        );

        /*
         * Inspect first session input.
         *
         * We do not execute inference yet.
         *
         * The actual Sana pipeline will later configure
         * its text, latent and image tensors.
         */
        MNN::Tensor* input =
                mInterpreter->getSessionInput(
                        mSession,
                        nullptr
                );

        if (input != nullptr) {

            /*
             * elementSize() is int in the MNN build
             * currently used by this project.
             *
             * Therefore %d is intentional.
             */
            LOGI(
                    "Model input detected. dims=%d elements=%d",
                    input->dimensions(),
                    input->elementSize()
            );

            const std::vector<int> shape =
                    input->shape();

            std::string shapeString;

            for (
                    size_t i = 0;
                    i < shape.size();
                    ++i
            ) {

                shapeString +=
                        std::to_string(
                                shape[i]
                        );

                if (
                        i + 1 <
                        shape.size()
                ) {

                    shapeString +=
                            " x ";
                }
            }

            LOGI(
                    "Input shape: %s",
                    shapeString.c_str()
            );
        } else {

            LOGI(
                    "MNN session has no unnamed input."
            );
        }

        return true;
    }


    void destroySession() {

        if (
                mInterpreter &&
                mSession
        ) {

            LOGI(
                    "Releasing MNN session."
            );

            mInterpreter->releaseSession(
                    mSession
            );

            mSession =
                    nullptr;
        }
    }


    void release() {

        std::lock_guard<std::mutex> lock(
                mMutex
        );

        destroySession();

        mInterpreter.reset();

        mModelBuffer.clear();

        mModelBuffer.shrink_to_fit();

        mBackend =
                "Not initialized";

        mLastError.clear();

        mCacheFile.clear();

        mInitialized =
                false;

        LOGI(
                "Sana native engine released."
        );
    }


    bool initialized() const {

        std::lock_guard<std::mutex> lock(
                mMutex
        );

        return mInitialized;
    }


    std::string backend() const {

        std::lock_guard<std::mutex> lock(
                mMutex
        );

        return mBackend;
    }


    std::string status() const {

        std::lock_guard<std::mutex> lock(
                mMutex
        );

        if (mInitialized) {

            return
                    "Ready | " +
                    mBackend;
        }

        if (!mLastError.empty()) {

            return
                    "Error | " +
                    mLastError;
        }

        return
                "Not initialized";
    }


private:

    void setError(
            const std::string& error
    ) {

        mLastError =
                error;
    }


private:

    /*
     * Protects interpreter/session state.
     */
    mutable std::mutex mMutex;

    /*
     * Persistent MNN interpreter.
     */
    std::unique_ptr<MNN::Interpreter>
            mInterpreter;

    /*
     * Persistent MNN session.
     */
    MNN::Session*
            mSession =
                    nullptr;

    /*
     * Model memory must remain alive while
     * the interpreter uses createFromBuffer().
     */
    std::vector<uint8_t>
            mModelBuffer;

    /*
     * Current backend.
     */
    std::string
            mBackend =
                    "Not initialized";

    /*
     * Last initialization error.
     */
    std::string
            mLastError;

    /*
     * OpenCL cache file.
     */
    std::string
            mCacheFile;

    /*
     * Initialization state.
     */
    bool
            mInitialized =
                    false;
};


/*
 * One persistent native engine.
 */
SanaEngine gEngine;

} // namespace


/*
 * ============================================================
 * JNI: INITIALIZE
 * ============================================================
 */
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv* env,
        jobject thiz,
        jobject assetManager,
        jstring modelAsset,
        jstring cachePath,
        jboolean preferOpenCl,
        jint cpuThreads
) {

    if (assetManager == nullptr) {

        LOGE(
                "AssetManager is null."
        );

        return JNI_FALSE;
    }

    if (modelAsset == nullptr) {

        LOGE(
                "Model asset string is null."
        );

        return JNI_FALSE;
    }

    /*
     * Convert model asset path from Java/Kotlin string.
     */
    const char* modelAssetChars =
            env->GetStringUTFChars(
                    modelAsset,
                    nullptr
            );

    if (modelAssetChars == nullptr) {

        LOGE(
                "Unable to read model asset string."
        );

        return JNI_FALSE;
    }

    std::string assetName(
            modelAssetChars
    );

    env->ReleaseStringUTFChars(
            modelAsset,
            modelAssetChars
    );

    /*
     * Convert cache path.
     */
    std::string cacheDirectory;

    if (cachePath != nullptr) {

        const char* cacheChars =
                env->GetStringUTFChars(
                        cachePath,
                        nullptr
                );

        if (cacheChars != nullptr) {

            cacheDirectory =
                    cacheChars;

            env->ReleaseStringUTFChars(
                    cachePath,
                    cacheChars
            );
        }
    }

    /*
     * Convert Java AssetManager to native
     * AAssetManager.
     */
    AAssetManager* nativeAssetManager =
            AAssetManager_fromJava(
                    env,
                    assetManager
            );

    if (nativeAssetManager == nullptr) {

        LOGE(
                "AAssetManager_fromJava failed."
        );

        return JNI_FALSE;
    }

    /*
     * Initialize persistent MNN engine.
     */
    const bool result =
            gEngine.initialize(
                    nativeAssetManager,
                    assetName,
                    cacheDirectory,
                    preferOpenCl == JNI_TRUE,
                    static_cast<int>(
                            cpuThreads
                    )
            );

    return result
           ? JNI_TRUE
           : JNI_FALSE;
}


/*
 * ============================================================
 * JNI: IS INITIALIZED
 * ============================================================
 */
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv* env,
        jobject thiz
) {

    return
            gEngine.initialized()
            ? JNI_TRUE
            : JNI_FALSE;
}


/*
 * ============================================================
 * JNI: GET BACKEND
 * ============================================================
 */
extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject thiz
) {

    const std::string backend =
            gEngine.backend();

    return env->NewStringUTF(
            backend.c_str()
    );
}


/*
 * ============================================================
 * JNI: GET STATUS
 * ============================================================
 */
extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject thiz
) {

    const std::string status =
            gEngine.status();

    return env->NewStringUTF(
            status.c_str()
    );
}


/*
 * ============================================================
 * JNI: RELEASE
 * ============================================================
 */
extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv* env,
        jobject thiz
) {

    gEngine.release();
}
