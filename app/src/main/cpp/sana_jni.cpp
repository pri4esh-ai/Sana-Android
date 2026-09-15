#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

class SanaEngine {

public:

    SanaEngine() = default;

    ~SanaEngine() {
        release();
    }

    bool initialize(
            const std::string& modelPath,
            const std::string& cachePath,
            bool preferOpenCl,
            int cpuThreads
    ) {

        std::lock_guard<std::mutex> lock(mMutex);

        if (mInitialized) {
            return true;
        }

        if (modelPath.empty()) {
            setError("Model path is empty.");
            return false;
        }

        LOGI("========================================");
        LOGI("SANA MNN ENGINE");
        LOGI("========================================");

        LOGI(
                "Model: %s",
                modelPath.c_str()
        );

        mInterpreter.reset(
                MNN::Interpreter::createFromFile(
                        modelPath.c_str()
                )
        );

        if (!mInterpreter) {

            setError(
                    "MNN createFromFile failed."
            );

            return false;
        }

        if (!cachePath.empty()) {

            mCacheFile =
                    cachePath +
                    "/sana_mnn_gpu.cache";

            mInterpreter->setCacheFile(
                    mCacheFile.c_str()
            );
        }

        if (preferOpenCl) {

            LOGI("Trying OpenCL / Precision_Low");

            if (createSession(
                    MNN_FORWARD_OPENCL,
                    1,
                    true
            )) {

                mBackend =
                        "OpenCL / FP16 Low Precision";

                mInitialized =
                        true;

                return true;
            }

            destroySession();

            LOGI(
                    "OpenCL unavailable. Trying CPU."
            );
        }

        const int threads =
                std::max(
                        1,
                        std::min(
                                cpuThreads,
                                8
                        )
                );

        if (!createSession(
                MNN_FORWARD_CPU,
                threads,
                true
        )) {

            destroySession();

            mInterpreter.reset();

            setError(
                    "Unable to create MNN session."
            );

            return false;
        }

        mBackend =
                "ARM CPU / Low Precision";

        mInitialized =
                true;

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

        config.type =
                backend;

        config.numThread =
                backend == MNN_FORWARD_CPU
                ? threads
                : 1;

        MNN::BackendConfig backendConfig;

        if (lowPrecision) {

            backendConfig.precision =
                    MNN::BackendConfig::Precision_Low;

            backendConfig.memory =
                    MNN::BackendConfig::Memory_Low;

            config.backendConfig =
                    &backendConfig;
        }

        MNN::Session* session =
                mInterpreter->createSession(
                        config
                );

        if (!session) {
            return false;
        }

        mSession =
                session;

        return true;
    }


    void destroySession() {

        if (
                mInterpreter &&
                mSession
        ) {

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

        mBackend =
                "Not initialized";

        mLastError.clear();

        mCacheFile.clear();

        mInitialized =
                false;
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

        return "Not initialized";
    }


private:

    void setError(
            const std::string& error
    ) {

        mLastError =
                error;

        LOGE(
                "%s",
                error.c_str()
        );
    }


private:

    mutable std::mutex mMutex;

    std::unique_ptr<MNN::Interpreter>
            mInterpreter;

    MNN::Session*
            mSession = nullptr;

    std::string
            mBackend =
                    "Not initialized";

    std::string
            mLastError;

    std::string
            mCacheFile;

    bool
            mInitialized = false;
};


SanaEngine gEngine;


/*
 * Convert tensor shape to text.
 */
std::string shapeString(
        const std::vector<int>& shape
) {

    std::ostringstream out;

    out << "[";

    for (
            size_t i = 0;
            i < shape.size();
            ++i
    ) {

        out << shape[i];

        if (
                i + 1 <
                shape.size()
        ) {

            out << ", ";
        }
    }

    out << "]";

    return out.str();
}


/*
 * Fill a floating point input tensor with zeros.
 *
 * Sana exported inputs are floating point tensors.
 */
bool fillFloatInput(
        MNN::Interpreter* interpreter,
        MNN::Session* session,
        MNN::Tensor* input,
        std::string& error
) {

    if (!input) {

        error =
                "Input tensor is null.";

        return false;
    }

    MNN::Tensor host(
            input,
            MNN::Tensor::CAFFE
    );

    const int elements =
            host.elementSize();

    if (elements <= 0) {

        error =
                "Input tensor has zero elements.";

        return false;
    }

    float* data =
            host.host<float>();

    if (!data) {

        error =
                "Unable to access host tensor.";

        return false;
    }

    std::fill(
            data,
            data + elements,
            0.0f
    );

    /*
     * The timestep input is normally a scalar.
     *
     * A zero timestep is intentionally used for this
     * structural inference test. This does NOT generate
     * an image yet.
     */
    interpreter->resizeSession(
            session
    );

    input->copyFromHostTensor(
            &host
    );

    return true;
}


/*
 * Run one MNN model with dummy tensors.
 */
std::string testSingleModel(
        const std::string& name,
        const std::string& path,
        const std::string& cachePath,
        bool preferOpenCl
) {

    std::ostringstream result;

    result
            << "========================================\n"
            << name
            << "\n"
            << "========================================\n";

    LOGI(
            "Testing model: %s",
            name.c_str()
    );

    LOGI(
            "Path: %s",
            path.c_str()
    );

    if (path.empty()) {

        result
                << "FAIL: empty model path\n";

        return result.str();
    }

    std::unique_ptr<MNN::Interpreter> interpreter(
            MNN::Interpreter::createFromFile(
                    path.c_str()
            )
    );

    if (!interpreter) {

        result
                << "FAIL: MNN could not open model\n";

        return result.str();
    }

    std::string cacheFile;

    if (!cachePath.empty()) {

        cacheFile =
                cachePath +
                "/sana_test_" +
                name +
                ".cache";

        interpreter->setCacheFile(
                cacheFile.c_str()
        );
    }

    MNN::ScheduleConfig config;

    if (preferOpenCl) {

        config.type =
                MNN_FORWARD_OPENCL;

        config.numThread =
                1;

    } else {

        config.type =
                MNN_FORWARD_CPU;

        config.numThread =
                4;
    }

    MNN::BackendConfig backendConfig;

    backendConfig.precision =
            MNN::BackendConfig::Precision_Low;

    backendConfig.memory =
            MNN::BackendConfig::Memory_Low;

    config.backendConfig =
            &backendConfig;

    MNN::Session* session =
            interpreter->createSession(
                    config
            );

    std::string backendName;

    if (session) {

        backendName =
                preferOpenCl
                ? "OpenCL / FP16"
                : "CPU / Low Precision";

    } else if (preferOpenCl) {

        LOGI(
                "%s: OpenCL failed. Trying CPU.",
                name.c_str()
        );

        config.type =
                MNN_FORWARD_CPU;

        config.numThread =
                4;

        session =
                interpreter->createSession(
                        config
                );

        backendName =
                "ARM CPU / Low Precision";
    }

    if (!session) {

        result
                << "FAIL: session creation failed\n";

        return result.str();
    }

    result
            << "Backend: "
            << backendName
            << "\n";

    /*
     * Get every model input.
     *
     * Transformer:
     *   hidden_states
     *   encoder_hidden_states
     *   timestep
     *
     * VAE:
     *   latent sample
     *
     * We discover the actual exported names/shapes
     * instead of hardcoding them.
     */
    auto inputs =
            interpreter->getSessionInputAll(
                    session
            );

    if (inputs.empty()) {

        result
                << "FAIL: no input tensors\n";

        interpreter->releaseSession(
                session
        );

        return result.str();
    }

    result
            << "Inputs: "
            << inputs.size()
            << "\n\n";

    for (const auto& item : inputs) {

        const std::string& inputName =
                item.first;

        MNN::Tensor* input =
                item.second;

        result
                << "Input: "
                << inputName
                << "\n";

        result
                << "Shape: "
                << shapeString(
                        input->shape()
                )
                << "\n";

        result
                << "Elements: "
                << input->elementSize()
                << "\n";

        std::string error;

        if (!fillFloatInput(
                interpreter.get(),
                session,
                input,
                error
        )) {

            result
                    << "FAIL filling input: "
                    << error
                    << "\n";

            interpreter->releaseSession(
                    session
            );

            return result.str();
        }
    }

    /*
     * Resize after all tensor information is known.
     */
    interpreter->resizeSession(
            session
    );

    result
            << "\nRunning inference...\n";

    const auto start =
            std::chrono::steady_clock::now();

    MNN::ErrorCode errorCode =
            interpreter->runSession(
                    session
            );

    const auto end =
            std::chrono::steady_clock::now();

    const double milliseconds =
            std::chrono::duration<double, std::milli>(
                    end - start
            ).count();

    result
            << "Time: "
            << milliseconds
            << " ms\n";

    result
            << "Error code: "
            << static_cast<int>(
                    errorCode
            )
            << "\n";

    if (
            errorCode !=
            MNN::NO_ERROR
    ) {

        result
                << "FAIL: MNN inference error\n";

        interpreter->releaseSession(
                session
        );

        return result.str();
    }

    /*
     * Inspect outputs.
     */
    auto outputs =
            interpreter->getSessionOutputAll(
                    session
            );

    result
            << "Outputs: "
            << outputs.size()
            << "\n";

    for (const auto& item : outputs) {

        MNN::Tensor* output =
                item.second;

        result
                << "Output: "
                << item.first
                << "\n";

        result
                << "Shape: "
                << shapeString(
                        output->shape()
                )
                << "\n";

        result
                << "Elements: "
                << output->elementSize()
                << "\n";
    }

    result
            << "\nPASS: model executed\n";

    /*
     * Very important for this test:
     *
     * Transformer is released before VAE starts.
     *
     * This avoids unnecessarily keeping both huge
     * model sessions alive at once.
     */
    interpreter->releaseSession(
            session
    );

    interpreter.reset();

    return result.str();
}


/*
 * Test Transformer then VAE sequentially.
 */
std::string testModels(
        const std::string& transformerPath,
        const std::string& vaePath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    std::ostringstream result;

    result
            << "SANA 0.6B / 512 MODEL TEST\n\n";

    result
            << "Transformer file:\n"
            << transformerPath
            << "\n\n";

    result
            << "VAE file:\n"
            << vaePath
            << "\n\n";

    /*
     * Transformer first.
     */
    result
            << testSingleModel(
                    "transformer",
                    transformerPath,
                    cachePath,
                    preferOpenCl
            );

    result
            << "\n\n";

    /*
     * VAE second.
     */
    result
            << testSingleModel(
                    "vae_decoder",
                    vaePath,
                    cachePath,
                    preferOpenCl
            );

    result
            << "\n\n========================================\n"
            << "TEST COMPLETE\n"
            << "========================================\n";

    return result.str();
}

} // namespace


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv* env,
        jobject,
        jobject assetManager,
        jstring modelAsset,
        jstring cachePath,
        jboolean preferOpenCl,
        jint cpuThreads
) {

    /*
     * Kept for compatibility with the existing engine.
     *
     * The new model-test UI does not use this method.
     */
    return JNI_FALSE;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv*,
        jobject
) {

    return gEngine.initialized()
           ? JNI_TRUE
           : JNI_FALSE;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject
) {

    const std::string value =
            gEngine.backend();

    return env->NewStringUTF(
            value.c_str()
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject
) {

    const std::string value =
            gEngine.status();

    return env->NewStringUTF(
            value.c_str()
    );
}


extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject
) {

    gEngine.release();
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestModels(
        JNIEnv* env,
        jobject,
        jstring transformerPath,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl
) {

    if (
            transformerPath == nullptr ||
            vaePath == nullptr
    ) {

        return env->NewStringUTF(
                "FAIL: model path is null."
        );
    }

    const char* transformerChars =
            env->GetStringUTFChars(
                    transformerPath,
                    nullptr
            );

    const char* vaeChars =
            env->GetStringUTFChars(
                    vaePath,
                    nullptr
            );

    const char* cacheChars =
            cachePath != nullptr
            ? env->GetStringUTFChars(
                    cachePath,
                    nullptr
            )
            : nullptr;

    if (
            transformerChars == nullptr ||
            vaeChars == nullptr
    ) {

        if (transformerChars) {
            env->ReleaseStringUTFChars(
                    transformerPath,
                    transformerChars
            );
        }

        if (vaeChars) {
            env->ReleaseStringUTFChars(
                    vaePath,
                    vaeChars
            );
        }

        if (cacheChars) {
            env->ReleaseStringUTFChars(
                    cachePath,
                    cacheChars
            );
        }

        return env->NewStringUTF(
                "FAIL: unable to read model paths."
        );
    }

    std::string transformer(
            transformerChars
    );

    std::string vae(
            vaeChars
    );

    std::string cache;

    if (cacheChars) {

        cache =
                cacheChars;
    }

    env->ReleaseStringUTFChars(
            transformerPath,
            transformerChars
    );

    env->ReleaseStringUTFChars(
            vaePath,
            vaeChars
    );

    if (cacheChars) {

        env->ReleaseStringUTFChars(
                cachePath,
                cacheChars
        );
    }

    /*
     * Run synchronously.
     *
     * Kotlin calls this from a background thread.
     */
    const std::string output =
            testModels(
                    transformer,
                    vae,
                    cache,
                    preferOpenCl == JNI_TRUE
            );

    return env->NewStringUTF(
            output.c_str()
    );
}
