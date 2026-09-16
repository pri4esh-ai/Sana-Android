#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <cstring>
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


/* ============================================================
 * SAFE STRING HELPERS
 * ============================================================ */

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


/* ============================================================
 * SANA ENGINE
 * ============================================================ */

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

        std::lock_guard<std::mutex> lock(
                mMutex
        );

        if (mInitialized) {

            return true;
        }


        if (modelPath.empty()) {

            setError(
                    "Model path is empty."
            );

            return false;
        }


        LOGI(
                "Loading Sana model: %s",
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


        MNN::ScheduleConfig config;

        MNN::BackendConfig backendConfig;


        backendConfig.precision =
                MNN::BackendConfig::Precision_Low;


        backendConfig.memory =
                MNN::BackendConfig::Memory_Low;


        config.backendConfig =
                &backendConfig;


        /*
         * OpenCL first.
         */

        if (preferOpenCl) {

            LOGI(
                    "Trying OpenCL / FP16..."
            );


            config.type =
                    MNN_FORWARD_OPENCL;


            config.numThread =
                    1;


            mSession =
                    mInterpreter->createSession(
                            config
                    );


            if (mSession) {

                mBackend =
                        "OpenCL / FP16";

                mInitialized =
                        true;

                LOGI(
                        "OpenCL session created."
                );

                return true;
            }


            LOGE(
                    "OpenCL session creation failed."
            );
        }


        /*
         * CPU fallback.
         */

        LOGI(
                "Trying CPU fallback..."
        );


        config.type =
                MNN_FORWARD_CPU;


        config.numThread =
                std::max(
                        1,
                        std::min(
                                cpuThreads,
                                8
                        )
                );


        mSession =
                mInterpreter->createSession(
                        config
                );


        if (!mSession) {

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


    void release() {

        std::lock_guard<std::mutex> lock(
                mMutex
        );


        if (
                mInterpreter &&
                mSession
        ) {

            mInterpreter->releaseSession(
                    mSession
            );
        }


        mSession =
                nullptr;


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
            mInitialized =
                    false;
};


SanaEngine gEngine;


/* ============================================================
 * SAFE INPUT INITIALIZATION
 *
 * IMPORTANT:
 *
 * MNN documentation requires:
 *
 * resizeTensor if changing dimensions
 *       ↓
 * resizeSession
 *       ↓
 * create/read host tensor
 *       ↓
 * copyFromHostTensor
 *       ↓
 * runSession
 *
 * We do NOT change Sana's exported dimensions here.
 * ============================================================ */

bool prepareInput(
        MNN::Tensor* input,
        const std::string& name,
        std::string& error
) {

    if (!input) {

        error =
                "Input tensor is null.";

        return false;
    }


    const std::vector<int> shape =
            input->shape();


    const int elements =
            input->elementSize();


    LOGI(
            "Preparing input: %s",
            name.c_str()
    );


    LOGI(
            "Shape: %s",
            shapeString(shape).c_str()
    );


    LOGI(
            "Elements: %d",
            elements
    );


    if (elements <= 0) {

        error =
                "Input tensor has zero elements.";

        return false;
    }


    /*
     * Create host tensor only AFTER resizeSession().
     *
     * This is the safe MNN pattern.
     */

    std::unique_ptr<MNN::Tensor> host(
            new MNN::Tensor(
                    input,
                    MNN::Tensor::CAFFE
            )
    );


    if (!host) {

        error =
                "Unable to allocate host tensor.";

        return false;
    }


    const int hostElements =
            host->elementSize();


    if (hostElements <= 0) {

        error =
                "Host tensor has zero elements.";

        return false;
    }


    float* data =
            host->host<float>();


    if (!data) {

        error =
                "Host tensor data is null.";

        return false;
    }


    /*
     * Zero initialization is enough for the
     * structural runtime test.
     */

    std::fill(
            data,
            data + hostElements,
            0.0f
    );


    /*
     * Give timestep a valid non-zero value.
     *
     * This is only for graph execution testing.
     */

    if (
            name == "timestep" &&
            hostElements > 0
    ) {

        data[0] =
                1.0f;
    }


    /*
     * Copy host → device.
     */

    const bool copied =
            input->copyFromHostTensor(
                    host.get()
            );


    if (!copied) {

        /*
         * CPU tensors can return false because
         * no device copy is required.
         *
         * Do not treat this alone as fatal.
         */

        LOGI(
                "copyFromHostTensor returned false for %s",
                name.c_str()
        );
    }


    return true;
}


/* ============================================================
 * SINGLE MODEL TEST
 * ============================================================ */

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
            "========================================"
    );


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


    /*
     * Load model.
     */

    std::unique_ptr<MNN::Interpreter>
            interpreter(
                    MNN::Interpreter::createFromFile(
                            path.c_str()
                    )
            );


    if (!interpreter) {

        result
                << "FAIL: MNN could not open model\n";

        return result.str();
    }


    LOGI(
            "%s loaded successfully.",
            name.c_str()
    );


    /*
     * GPU cache.
     */

    if (!cachePath.empty()) {

        const std::string cacheFile =
                cachePath +
                "/sana_test_" +
                name +
                ".cache";


        interpreter->setCacheFile(
                cacheFile.c_str()
        );
    }


    /*
     * Session configuration.
     */

    MNN::ScheduleConfig config;

    MNN::BackendConfig backendConfig;


    backendConfig.precision =
            MNN::BackendConfig::Precision_Low;


    backendConfig.memory =
            MNN::BackendConfig::Memory_Low;


    config.backendConfig =
            &backendConfig;


    config.type =
            preferOpenCl
            ? MNN_FORWARD_OPENCL
            : MNN_FORWARD_CPU;


    config.numThread =
            preferOpenCl
            ? 1
            : 4;


    /*
     * Create session.
     */

    LOGI(
            "Creating session..."
    );


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

        /*
         * OpenCL failed.
         * Try CPU.
         */

        LOGE(
                "%s: OpenCL session failed.",
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
     * Get model inputs.
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


    /*
     * IMPORTANT:
     *
     * Session created with the model's
     * exported dimensions.
     *
     * We don't change any dimensions.
     *
     * Just perform the required MNN
     * allocation/resize pass.
     */

    LOGI(
            "Calling resizeSession..."
    );


    interpreter->resizeSession(
            session
    );


    /*
     * Re-fetch inputs after resize.
     *
     * This avoids keeping stale tensor
     * pointers if MNN internally updates
     * tensors during resize.
     */

    inputs =
            interpreter->getSessionInputAll(
                    session
            );


    if (inputs.empty()) {

        result
                << "FAIL: inputs disappeared after resize\n";


        interpreter->releaseSession(
                session
        );


        return result.str();
    }


    /*
     * Describe and prepare every input.
     */

    for (const auto& item : inputs) {

        const std::string& inputName =
                item.first;


        MNN::Tensor* input =
                item.second;


        if (!input) {

            result
                    << "FAIL: null input: "
                    << inputName
                    << "\n";


            interpreter->releaseSession(
                    session
            );


            return result.str();
        }


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


        if (!prepareInput(
                input,
                inputName,
                error
        )) {

            result
                    << "FAIL preparing input: "
                    << error
                    << "\n";


            interpreter->releaseSession(
                    session
            );


            return result.str();
        }
    }


    /*
     * Run inference.
     */

    result
            << "\nRunning inference...\n";


    LOGI(
            "Running %s...",
            name.c_str()
    );


    const auto start =
            std::chrono::steady_clock::now();


    const MNN::ErrorCode errorCode =
            interpreter->runSession(
                    session
            );


    const auto end =
            std::chrono::steady_clock::now();


    const double milliseconds =
            std::chrono::duration<
                    double,
                    std::milli
            >(
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


        LOGE(
                "%s inference failed: %d",
                name.c_str(),
                static_cast<int>(
                        errorCode
                )
        );


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


        if (!output) {

            result
                    << "Output: "
                    << item.first
                    << "\n"
                    << "ERROR: null output\n";


            continue;
        }


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


    LOGI(
            "%s PASS",
            name.c_str()
    );


    /*
     * Release this huge model completely
     * before the next model is loaded.
     */

    interpreter->releaseSession(
            session
    );


    session =
            nullptr;


    interpreter.reset();


    return result.str();
}


/* ============================================================
 * TEST TRANSFORMER + VAE SEQUENTIALLY
 * ============================================================ */

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
     * Transformer.
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
     * VAE.
     *
     * Transformer has already been completely
     * released before this starts.
     */

    result
            << testSingleModel(
                    "vae_decoder",
                    vaePath,
                    cachePath,
                    preferOpenCl
            );


    result
            << "\n\n";


    result
            << "========================================\n"
            << "TEST COMPLETE\n"
            << "========================================\n";


    return result.str();
}


} // namespace


/* ============================================================
 * EXISTING JNI API
 * ============================================================ */

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv*,
        jobject,
        jobject,
        jstring,
        jstring,
        jboolean,
        jint
) {

    /*
     * Kept for compatibility with the current Kotlin API.
     *
     * Current test UI uses nativeTestModels().
     */

    return JNI_FALSE;
}


/* ============================================================
 * nativeIsInitialized
 * ============================================================ */

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


/* ============================================================
 * nativeGetBackend
 * ============================================================ */

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


/* ============================================================
 * nativeGetStatus
 * ============================================================ */

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


/* ============================================================
 * nativeRelease
 * ============================================================ */

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject
) {

    gEngine.release();
}


/* ============================================================
 * nativeTestModels
 * ============================================================ */

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

    /*
     * Validate JNI arguments first.
     */

    if (
            transformerPath == nullptr ||
            vaePath == nullptr
    ) {

        return env->NewStringUTF(
                "FAIL: model path is null."
        );
    }


    /*
     * Read Transformer path.
     */

    const char* transformerChars =
            env->GetStringUTFChars(
                    transformerPath,
                    nullptr
            );


    /*
     * Read VAE path.
     */

    const char* vaeChars =
            env->GetStringUTFChars(
                    vaePath,
                    nullptr
            );


    /*
     * Read cache path.
     */

    const char* cacheChars =
            nullptr;


    if (cachePath != nullptr) {

        cacheChars =
                env->GetStringUTFChars(
                        cachePath,
                        nullptr
                );
    }


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


    /*
     * Copy JNI strings into normal C++ strings.
     */

    std::string transformer =
            transformerChars;


    std::string vae =
            vaeChars;


    std::string cache;


    if (cacheChars) {

        cache =
                cacheChars;
    }


    /*
     * Release JNI strings.
     */

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
     * Kotlin calls this from its Executor,
     * so the Android UI thread is not blocked.
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
