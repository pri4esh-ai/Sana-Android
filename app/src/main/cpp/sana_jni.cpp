#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


namespace {


/* ============================================================
 * SAFE STRING / SHAPE HELPERS
 * ============================================================ */

std::string shapeString(
        const std::vector<int>& shape
) {
    std::ostringstream out;

    out << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        out << shape[i];

        if (i + 1 < shape.size()) {
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

        std::lock_guard<std::mutex> lock(mMutex);

        if (mInitialized) {
            return true;
        }

        if (modelPath.empty()) {
            setError("Model path is empty.");
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

            /*
             * Cache disabled during the diagnostic
             * workflow would also be acceptable.
             *
             * Keep the existing API behavior here.
             */
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


        /* ----------------------------------------------------
         * OpenCL
         * ---------------------------------------------------- */

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


        /* ----------------------------------------------------
         * CPU fallback
         * ---------------------------------------------------- */

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
 * PREPARE ONE INPUT
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
     * Host tensor uses the model tensor's
     * existing dimensions.
     *
     * We do NOT resize the Sana model.
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
     * Structural runtime test.
     *
     * Zero values are intentionally used.
     */

    std::fill(
            data,
            data + hostElements,
            0.0f
    );


    /*
     * Timestep gets a non-zero value.
     */

    if (
            name == "timestep" &&
            hostElements > 0
    ) {

        data[0] =
                1.0f;
    }


    /*
     * Host -> device.
     */

    const bool copied =
            input->copyFromHostTensor(
                    host.get()
            );


    if (!copied) {

        LOGI(
                "copyFromHostTensor returned false for %s",
                name.c_str()
        );

        /*
         * Do not automatically fail here.
         *
         * MNN CPU tensors may not require a device
         * copy in the same way as GPU tensors.
         */
    }


    return true;
}


/* ============================================================
 * TRANSFORMER-ONLY TEST
 *
 * THIS FUNCTION NEVER LOADS THE VAE.
 * ============================================================ */

std::string testTransformerOnly(
        const std::string& transformerPath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    std::ostringstream result;


    result
            << "SANA 0.6B / 512 TRANSFORMER TEST\n\n";


    result
            << "Transformer file:\n"
            << transformerPath
            << "\n\n";


    if (transformerPath.empty()) {

        result
                << "FAIL: Transformer path is empty.\n";

        return result.str();
    }


    /* --------------------------------------------------------
     * Check file
     * -------------------------------------------------------- */

    struct stat fileStat {};

    if (
            stat(
                    transformerPath.c_str(),
                    &fileStat
            ) != 0
    ) {

        result
                << "FAIL: Transformer file does not exist.\n";

        return result.str();
    }


    if (fileStat.st_size <= 0) {

        result
                << "FAIL: Transformer file is empty.\n";

        return result.str();
    }


    result
            << "File size: "
            << static_cast<long long>(
                    fileStat.st_size
            )
            << " bytes\n\n";


    LOGI(
            "Transformer size: %lld bytes",
            static_cast<long long>(
                    fileStat.st_size
            )
    );


    /* --------------------------------------------------------
     * Load ONLY Transformer
     * -------------------------------------------------------- */

    LOGI(
            "Loading Transformer ONLY..."
    );


    std::unique_ptr<MNN::Interpreter>
            interpreter(
                    MNN::Interpreter::createFromFile(
                            transformerPath.c_str()
                    )
            );


    if (!interpreter) {

        result
                << "FAIL: MNN could not open Transformer.\n";

        return result.str();
    }


    result
            << "Transformer interpreter created.\n\n";


    LOGI(
            "Transformer interpreter created."
    );


    /*
     * IMPORTANT:
     *
     * Do not use GPU cache for this first diagnostic.
     *
     * A broken/stale cache should not interfere with
     * model validation.
     */

    (void) cachePath;


    /* --------------------------------------------------------
     * Session configuration
     * -------------------------------------------------------- */

    MNN::ScheduleConfig config;

    MNN::BackendConfig backendConfig;


    backendConfig.precision =
            MNN::BackendConfig::Precision_Low;


    backendConfig.memory =
            MNN::BackendConfig::Memory_Low;


    config.backendConfig =
            &backendConfig;


    MNN::Session* session =
            nullptr;


    std::string backendName;


    /* --------------------------------------------------------
     * OpenCL
     * -------------------------------------------------------- */

    if (preferOpenCl) {

        result
                << "Backend: OpenCL / FP16\n";


        LOGI(
                "Creating OpenCL / FP16 session..."
        );


        config.type =
                MNN_FORWARD_OPENCL;


        config.numThread =
                1;


        session =
                interpreter->createSession(
                        config
                );


        if (!session) {

            result
                    << "OpenCL session creation failed.\n"
                    << "Trying ARM CPU fallback...\n";


            LOGE(
                    "OpenCL session creation failed."
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

        } else {

            backendName =
                    "OpenCL / FP16";
        }

    } else {

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
                << "FAIL: MNN session creation failed.\n";

        return result.str();
    }


    /*
     * Update backend line if CPU fallback occurred.
     */

    result
            << "Actual backend: "
            << backendName
            << "\n\n";


    /* --------------------------------------------------------
     * Get model inputs
     * -------------------------------------------------------- */

    auto inputs =
            interpreter->getSessionInputAll(
                    session
            );


    result
            << "Inputs: "
            << inputs.size()
            << "\n\n";


    if (inputs.empty()) {

        result
                << "FAIL: Transformer has no inputs.\n";


        interpreter->releaseSession(
                session
        );


        return result.str();
    }


    /* --------------------------------------------------------
     * resizeSession
     *
     * We do NOT change dimensions.
     *
     * This lets MNN allocate the graph using the
     * exported Sana dimensions.
     * -------------------------------------------------------- */

    LOGI(
            "Calling resizeSession..."
    );


    interpreter->resizeSession(
            session
    );


    /*
     * Re-fetch tensors after resize.
     */

    inputs =
            interpreter->getSessionInputAll(
                    session
            );


    if (inputs.empty()) {

        result
                << "FAIL: No inputs after resizeSession.\n";


        interpreter->releaseSession(
                session
        );


        return result.str();
    }


    /* --------------------------------------------------------
     * Describe + prepare inputs
     * -------------------------------------------------------- */

    for (const auto& item : inputs) {

        const std::string& inputName =
                item.first;


        MNN::Tensor* input =
                item.second;


        if (!input) {

            result
                    << "FAIL: Null input: "
                    << inputName
                    << "\n";


            interpreter->releaseSession(
                    session
            );


            return result.str();
        }


        const std::vector<int> shape =
                input->shape();


        result
                << "Input: "
                << inputName
                << "\n";


        result
                << "Shape: "
                << shapeString(shape)
                << "\n";


        result
                << "Elements: "
                << input->elementSize()
                << "\n";


        std::string error;


        if (
                !prepareInput(
                        input,
                        inputName,
                        error
                )
        ) {

            result
                    << "FAIL preparing input: "
                    << error
                    << "\n";


            interpreter->releaseSession(
                    session
            );


            return result.str();
        }


        result
                << "\n";
    }


    /* --------------------------------------------------------
     * Verify expected Sana inputs
     * -------------------------------------------------------- */

    MNN::Tensor* encoderHiddenStates =
            interpreter->getSessionInput(
                    session,
                    "encoder_hidden_states"
            );


    MNN::Tensor* hiddenStates =
            interpreter->getSessionInput(
                    session,
                    "hidden_states"
            );


    MNN::Tensor* timestep =
            interpreter->getSessionInput(
                    session,
                    "timestep"
            );


    if (!encoderHiddenStates) {

        result
                << "WARNING: encoder_hidden_states "
                << "not found by name.\n";
    }


    if (!hiddenStates) {

        result
                << "WARNING: hidden_states "
                << "not found by name.\n";
    }


    if (!timestep) {

        result
                << "WARNING: timestep "
                << "not found by name.\n";
    }


    /* --------------------------------------------------------
     * Run inference
     * -------------------------------------------------------- */

    result
            << "\nRunning inference...\n";


    LOGI(
            "Running Transformer inference..."
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
            << "\n\n";


    if (
            errorCode !=
            MNN::NO_ERROR
    ) {

        result
                << "FAIL: Transformer inference error.\n";


        LOGE(
                "Transformer inference failed: %d",
                static_cast<int>(
                        errorCode
                )
        );


        interpreter->releaseSession(
                session
        );


        return result.str();
    }


    /* --------------------------------------------------------
     * Inspect outputs
     * -------------------------------------------------------- */

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
            << "\nPASS: Transformer executed successfully\n";


    LOGI(
            "Transformer PASS"
    );


    /* --------------------------------------------------------
     * RELEASE HUGE MODEL
     * -------------------------------------------------------- */

    interpreter->releaseSession(
            session
    );


    session =
            nullptr;


    interpreter.reset();


    LOGI(
            "Transformer interpreter released."
    );


    result
            << "\nTransformer released successfully.";


    return result.str();
}


/* ============================================================
 * ORIGINAL SINGLE MODEL TEST
 *
 * Kept for compatibility with nativeTestModels().
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


    if (path.empty()) {

        result
                << "FAIL: empty model path\n";

        return result.str();
    }


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


    /*
     * Allocate model using its exported dimensions.
     */

    interpreter->resizeSession(
            session
    );


    inputs =
            interpreter->getSessionInputAll(
                    session
            );


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


        if (
                !prepareInput(
                        input,
                        inputName,
                        error
                )
        ) {

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


    result
            << "\nRunning inference...\n";


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


        interpreter->releaseSession(
                session
        );


        return result.str();
    }


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


    interpreter->releaseSession(
            session
    );


    return result.str();
}


/* ============================================================
 * TEST TRANSFORMER + VAE SEQUENTIALLY
 *
 * Kept for the old API.
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


    result
            << testSingleModel(
                    "transformer",
                    transformerPath,
                    cachePath,
                    preferOpenCl
            );


    result
            << "\n\n";


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
 * nativeInitialize
 * ============================================================ */

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv* env,
        jobject,
        jobject,
        jstring modelAsset,
        jstring cachePath,
        jboolean preferOpenCl,
        jint cpuThreads
) {

    try {

        if (modelAsset == nullptr) {

            return JNI_FALSE;
        }


        const char* modelChars =
                env->GetStringUTFChars(
                        modelAsset,
                        nullptr
                );


        if (!modelChars) {

            return JNI_FALSE;
        }


        std::string model =
                modelChars;


        env->ReleaseStringUTFChars(
                modelAsset,
                modelChars
        );


        std::string cache;


        if (cachePath != nullptr) {

            const char* cacheChars =
                    env->GetStringUTFChars(
                            cachePath,
                            nullptr
                    );


            if (cacheChars) {

                cache =
                        cacheChars;


                env->ReleaseStringUTFChars(
                        cachePath,
                        cacheChars
                );
            }
        }


        return gEngine.initialize(
                model,
                cache,
                preferOpenCl == JNI_TRUE,
                static_cast<int>(
                        cpuThreads
                )
        )
               ? JNI_TRUE
               : JNI_FALSE;

    } catch (...) {

        return JNI_FALSE;
    }
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
 * NEW:
 * nativeTestTransformer
 *
 * Kotlin:
 *
 * external fun nativeTestTransformer(
 *     transformerPath: String,
 *     cachePath: String,
 *     preferOpenCl: Boolean
 * ): String
 *
 * ============================================================ */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
        JNIEnv* env,
        jobject,
        jstring transformerPath,
        jstring cachePath,
        jboolean preferOpenCl
) {

    try {

        if (transformerPath == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Transformer path is null."
            );
        }


        const char* transformerChars =
                env->GetStringUTFChars(
                        transformerPath,
                        nullptr
                );


        if (!transformerChars) {

            return env->NewStringUTF(
                    "FAIL: Unable to read Transformer path."
            );
        }


        std::string transformer =
                transformerChars;


        env->ReleaseStringUTFChars(
                transformerPath,
                transformerChars
        );


        std::string cache;


        if (cachePath != nullptr) {

            const char* cacheChars =
                    env->GetStringUTFChars(
                            cachePath,
                            nullptr
                    );


            if (cacheChars) {

                cache =
                        cacheChars;


                env->ReleaseStringUTFChars(
                        cachePath,
                        cacheChars
                );
            }
        }


        /*
         * IMPORTANT:
         *
         * This function calls ONLY:
         *
         * testTransformerOnly()
         *
         * No VAE path exists here.
         * No VAE interpreter is created.
         */

        const std::string output =
                testTransformerOnly(
                        transformer,
                        cache,
                        preferOpenCl == JNI_TRUE
                );


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (const std::exception& e) {

        std::string message =
                "C++ exception: ";

        message +=
                e.what();


        return env->NewStringUTF(
                message.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown native C++ exception."
        );
    }
}


/* ============================================================
 * ORIGINAL nativeTestModels
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

    try {

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


        std::string transformer =
                transformerChars;


        std::string vae =
                vaeChars;


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
         * Old API:
         *
         * Transformer first.
         * Transformer fully released.
         * VAE second.
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

    } catch (const std::exception& e) {

        std::string message =
                "C++ exception: ";

        message +=
                e.what();


        return env->NewStringUTF(
                message.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown native C++ exception."
        );
    }
}
