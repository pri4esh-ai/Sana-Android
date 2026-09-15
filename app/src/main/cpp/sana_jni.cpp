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
 * Important:
 * The interpreter and session stay alive between generations.
 *
 * This avoids:
 *
 * Generate 1
 *   -> load model
 *   -> create session
 *   -> inference
 *   -> destroy
 *
 * Instead:
 *
 * initialize once
 *   -> interpreter
 *   -> session
 *
 * Generate 1
 * Generate 2
 * Generate 3
 * ...
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
            LOGI("Sana engine already initialized.");
            return true;
        }

        if (assetManager == nullptr) {
            setError("AssetManager is null.");
            return false;
        }

        if (assetName.empty()) {
            setError("Model asset path is empty.");
            return false;
        }

        LOGI("========================================");
        LOGI("SANA MNN ENGINE INITIALIZATION");
        LOGI("========================================");

        LOGI("Model asset: %s", assetName.c_str());

        AAsset* asset = AAssetManager_open(
                assetManager,
                assetName.c_str(),
                AASSET_MODE_BUFFER
        );

        if (asset == nullptr) {
            setError(
                    "Unable to open model asset: " + assetName
            );

            LOGE(
                    "Could not open model asset: %s",
                    assetName.c_str()
            );

            return false;
        }

        const off_t assetLength = AAsset_getLength(asset);

        if (assetLength <= 0) {

            AAsset_close(asset);

            setError("Model asset is empty.");

            LOGE("Model asset is empty.");

            return false;
        }

        LOGI(
                "Model size: %.2f MB",
                static_cast<double>(assetLength) /
                (1024.0 * 1024.0)
        );

        /**
         * Keep the model bytes alive for the lifetime of the
         * interpreter. This makes createFromBuffer safe even
         * if the MNN build references the supplied buffer.
         */
        mModelBuffer.resize(
                static_cast<size_t>(assetLength)
        );

        const int64_t bytesRead = AAsset_read(
                asset,
                mModelBuffer.data(),
                static_cast<size_t>(assetLength)
        );

        AAsset_close(asset);

        if (bytesRead != assetLength) {

            mModelBuffer.clear();

            setError("Failed to read complete MNN model.");

            LOGE(
                    "Model read failed. Expected=%lld Actual=%lld",
                    static_cast<long long>(assetLength),
                    static_cast<long long>(bytesRead)
            );

            return false;
        }

        LOGI("Model loaded into native memory.");

        /**
         * Create MNN interpreter from memory.
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
                    "MNN Interpreter::createFromBuffer failed."
            );

            LOGE(
                    "MNN Interpreter creation failed."
            );

            return false;
        }

        LOGI("MNN interpreter created.");

        /**
         * GPU kernel/cache persistence.
         *
         * This can significantly reduce repeated OpenCL
         * initialization/tuning.
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

        /**
         * Prefer OpenCL.
         *
         * If OpenCL cannot create a session, we fall back
         * automatically to CPU.
         */
        if (preferOpenCl) {

            if (createSession(
                    MNN_FORWARD_OPENCL,
                    1,
                    true
            )) {

                mBackend = "OpenCL / FP16";
                mInitialized = true;

                LOGI(
                        "Sana backend: OpenCL / low precision"
                );

                return true;
            }

            LOGI(
                    "OpenCL session unavailable. "
                    "Falling back to ARM CPU."
            );

            destroySession();
        }

        /**
         * CPU fallback.
         */
        const int safeThreads =
                std::max(
                        1,
                        std::min(cpuThreads, 8)
                );

        if (!createSession(
                MNN_FORWARD_CPU,
                safeThreads,
                true
        )) {

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
                "ARM CPU / FP16-if-supported";

        mInitialized = true;

        LOGI(
                "Sana backend: ARM CPU / low precision"
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
            return false;
        }

        MNN::ScheduleConfig config;

        config.type = backend;

        /**
         * For CPU this is thread count.
         *
         * For GPU the field is used differently by MNN,
         * so we keep it conservative.
         */
        config.numThread =
                backend == MNN_FORWARD_CPU
                ? threads
                : 1;

        /**
         * Low precision:
         *
         * CPU:
         *   lets MNN use FP16 where the device/backend supports it.
         *
         * OpenCL:
         *   allows the backend to use lower precision paths.
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
                mInterpreter->createSession(config);

        if (session == nullptr) {

            LOGE(
                    "MNN createSession failed. backend=%d",
                    static_cast<int>(backend)
            );

            return false;
        }

        mSession = session;

        LOGI("MNN session created.");

        /**
         * Get model input/output information.
         *
         * We don't execute anything yet because the actual
         * Sana pipeline will determine tensor shapes.
         */
        MNN::Tensor* input =
                mInterpreter->getSessionInput(
                        mSession,
                        nullptr
                );

        if (input != nullptr) {

            LOGI(
                    "Model input detected. dims=%d elements=%zu",
                    input->dimensions(),
                    input->elementSize()
            );

            const std::vector<int> shape =
                    input->shape();

            std::string shapeString;

            for (size_t i = 0; i < shape.size(); ++i) {

                shapeString +=
                        std::to_string(shape[i]);

                if (i + 1 < shape.size()) {
                    shapeString += " x ";
                }
            }

            LOGI(
                    "Input shape: %s",
                    shapeString.c_str()
            );
        }

        return true;
    }


    void destroySession() {

        if (mInterpreter && mSession) {

            mInterpreter->releaseSession(
                    mSession
            );

            mSession = nullptr;
        }
    }


    void release() {

        std::lock_guard<std::mutex> lock(mMutex);

        destroySession();

        mInterpreter.reset();

        mModelBuffer.clear();
        mModelBuffer.shrink_to_fit();

        mBackend = "Not initialized";

        mLastError.clear();

        mInitialized = false;

        LOGI(
                "Sana native engine released."
        );
    }


    bool initialized() const {
        return mInitialized;
    }


    std::string backend() const {

        std::lock_guard<std::mutex> lock(mMutex);

        return mBackend;
    }


    std::string status() const {

        std::lock_guard<std::mutex> lock(mMutex);

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

        return "Not initialized";
    }


private:

    void setError(
            const std::string& error
    ) {

        mLastError = error;
    }


private:

    mutable std::mutex mMutex;

    std::unique_ptr<MNN::Interpreter>
            mInterpreter;

    MNN::Session*
            mSession = nullptr;

    /**
     * Model memory is intentionally retained.
     */
    std::vector<uint8_t>
            mModelBuffer;

    std::string
            mBackend = "Not initialized";

    std::string
            mLastError;

    std::string
            mCacheFile;

    bool
            mInitialized = false;
};


/**
 * One persistent native engine for the entire application.
 */
SanaEngine gEngine;

} // namespace


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

        LOGE("AssetManager is null.");

        return JNI_FALSE;
    }

    if (modelAsset == nullptr) {

        LOGE("Model asset string is null.");

        return JNI_FALSE;
    }

    const char* modelAssetChars =
            env->GetStringUTFChars(
                    modelAsset,
                    nullptr
            );

    if (modelAssetChars == nullptr) {
        return JNI_FALSE;
    }

    std::string assetName(
            modelAssetChars
    );

    env->ReleaseStringUTFChars(
            modelAsset,
            modelAssetChars
    );


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


    const bool result =
            gEngine.initialize(
                    nativeAssetManager,
                    assetName,
                    cacheDirectory,
                    preferOpenCl == JNI_TRUE,
                    static_cast<int>(cpuThreads)
            );

    return result
           ? JNI_TRUE
           : JNI_FALSE;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv* env,
        jobject thiz
) {

    return gEngine.initialized()
           ? JNI_TRUE
           : JNI_FALSE;
}


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


extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv* env,
        jobject thiz
) {

    gEngine.release();
}
