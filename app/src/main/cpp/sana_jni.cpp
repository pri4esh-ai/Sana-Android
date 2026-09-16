#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>


// ============================================================
// Logging
// ============================================================

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


// ============================================================
// Utility
// ============================================================

static std::string shapeString(
        const std::vector<int>& shape
) {
    std::ostringstream ss;

    ss << "[";

    for (size_t i = 0; i < shape.size(); ++i) {

        if (i > 0) {
            ss << ", ";
        }

        ss << shape[i];
    }

    ss << "]";

    return ss.str();
}


static std::string jstringToString(
        JNIEnv* env,
        jstring value
) {
    if (value == nullptr) {
        return "";
    }

    const char* chars =
            env->GetStringUTFChars(
                    value,
                    nullptr
            );

    if (chars == nullptr) {
        return "";
    }

    std::string result(chars);

    env->ReleaseStringUTFChars(
            value,
            chars
    );

    return result;
}


// ============================================================
// Persistent Sana Engine
// ============================================================

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

        (void) cachePath;

        std::lock_guard<std::mutex> lock(mMutex);

        releaseLocked();


        struct stat fileInfo {};

        if (
                stat(
                        modelPath.c_str(),
                        &fileInfo
                ) != 0
        ) {

            mStatus =
                    "Model file does not exist:\n" +
                    modelPath;

            return false;
        }


        mInterpreter.reset(
                MNN::Interpreter::createFromFile(
                        modelPath.c_str()
                )
        );


        if (!mInterpreter) {

            mStatus =
                    "Failed to create MNN interpreter.";

            return false;
        }


        MNN::ScheduleConfig config{};


        if (preferOpenCl) {

            config.type =
                    MNN_FORWARD_OPENCL;

            config.numThread =
                    cpuThreads;

            config.mode =
                    MNN_GPU_TUNING_FAST;

        } else {

            config.type =
                    MNN_FORWARD_CPU;

            config.numThread =
                    cpuThreads;

            config.mode =
                    MNN_GPU_TUNING_NONE;
        }


        MNN::BackendConfig backendConfig{};

        backendConfig.precision =
                MNN::BackendConfig::Precision_Low;

        config.backendConfig =
                &backendConfig;


        mSession =
                mInterpreter->createSession(
                        config
                );


        if (!mSession) {

            mStatus =
                    "Failed to create MNN session.";

            mInterpreter.reset();

            return false;
        }


        mInitialized = true;


        if (preferOpenCl) {

            mBackend =
                    "OpenCL / FP16";

        } else {

            mBackend =
                    "CPU";
        }


        mStatus =
                "Sana engine initialized successfully.";

        return true;
    }


    bool isInitialized() {

        std::lock_guard<std::mutex> lock(mMutex);

        return mInitialized;
    }


    std::string backend() {

        std::lock_guard<std::mutex> lock(mMutex);

        return mBackend;
    }


    std::string status() {

        std::lock_guard<std::mutex> lock(mMutex);

        return mStatus;
    }


    void release() {

        std::lock_guard<std::mutex> lock(mMutex);

        releaseLocked();
    }


private:

    void releaseLocked() {

        if (mInterpreter) {

            if (mSession) {

                mInterpreter->releaseSession(
                        mSession
                );

                mSession = nullptr;
            }

            mInterpreter.reset();
        }


        mInitialized = false;

        mBackend.clear();

        mStatus =
                "Sana engine released.";
    }


private:

    std::mutex mMutex;

    std::unique_ptr<MNN::Interpreter>
            mInterpreter;

    MNN::Session* mSession =
            nullptr;

    bool mInitialized =
            false;

    std::string mBackend;

    std::string mStatus =
            "Sana engine not initialized.";
};


static SanaEngine gEngine;


// ============================================================
// Generic Transformer input preparation
//
// This function is intentionally preserved for the Transformer
// diagnostic that already works.
// ============================================================

static bool prepareInput(
        MNN::Interpreter* interpreter,
        MNN::Session* session,
        MNN::Tensor* input,
        const std::string& inputName
) {

    (void) interpreter;
    (void) session;


    if (!input) {

        LOGE(
                "Input tensor is null: %s",
                inputName.c_str()
        );

        return false;
    }


    const std::vector<int> shape =
            input->shape();


    LOGI(
            "Preparing input %s shape=%s",
            inputName.c_str(),
            shapeString(shape).c_str()
    );


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE
    );


    float* hostData =
            hostTensor.host<float>();


    if (!hostData) {

        LOGE(
                "Host tensor allocation failed: %s",
                inputName.c_str()
        );

        return false;
    }


    const size_t elementCount =
            hostTensor.elementSize();


    std::fill(
            hostData,
            hostData + elementCount,
            0.0f
    );


    if (inputName == "timestep") {

        if (elementCount > 0) {

            hostData[0] =
                    1.0f;
        }
    }


    const bool copied =
            input->copyFromHostTensor(
                    &hostTensor
            );


    if (!copied && input->deviceId() != 0) {

        LOGE(
                "copyFromHostTensor failed for %s",
                inputName.c_str()
        );

        return false;
    }


    return true;
}


// ============================================================
// Generic single-model diagnostic
//
// This preserves the Transformer diagnostic behavior.
// ============================================================

static std::string testSingleModel(
        const std::string& modelName,
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    (void) cachePath;

    std::ostringstream result;


    result
            << modelName
            << "\n"
            << "===\n";


    // --------------------------------------------------------
    // Check model file
    // --------------------------------------------------------

    struct stat fileInfo {};

    if (
            stat(
                    modelPath.c_str(),
                    &fileInfo
            ) != 0
    ) {

        result
                << "FAIL: Model file does not exist.\n"
                << modelPath;

        return result.str();
    }


    result
            << "Model file:\n"
            << modelPath
            << "\n\n";


    result
            << "File size: "
            << static_cast<long long>(
                    fileInfo.st_size
            )
            << " bytes\n\n";


    // --------------------------------------------------------
    // Create interpreter
    // --------------------------------------------------------

    std::shared_ptr<MNN::Interpreter>
            interpreter(
                    MNN::Interpreter::createFromFile(
                            modelPath.c_str()
                    )
            );


    if (!interpreter) {

        result
                << "FAIL: Could not create MNN interpreter.";

        return result.str();
    }


    result
            << "Interpreter created.\n\n";


    // --------------------------------------------------------
    // Session configuration
    // --------------------------------------------------------

    MNN::ScheduleConfig config{};


    if (preferOpenCl) {

        config.type =
                MNN_FORWARD_OPENCL;

        config.numThread =
                4;

        config.mode =
                MNN_GPU_TUNING_FAST;

    } else {

        config.type =
                MNN_FORWARD_CPU;

        config.numThread =
                4;

        config.mode =
                MNN_GPU_TUNING_NONE;
    }


    MNN::BackendConfig backendConfig{};

    backendConfig.precision =
            MNN::BackendConfig::Precision_Low;

    config.backendConfig =
            &backendConfig;


    MNN::Session* session =
            interpreter->createSession(
                    config
            );


    if (!session) {

        result
                << "FAIL: Could not create MNN session.";

        return result.str();
    }


    if (preferOpenCl) {

        result
                << "Backend: OpenCL / FP16\n\n";

    } else {

        result
                << "Backend: CPU\n\n";
    }


    // --------------------------------------------------------
    // Prepare session
    // --------------------------------------------------------

    interpreter->resizeSession(
            session
    );


    // --------------------------------------------------------
    // Get inputs
    // --------------------------------------------------------

    std::map<std::string, MNN::Tensor*> inputs =
            interpreter->getSessionInputAll(
                    session
            );


    result
            << "Inputs: "
            << inputs.size()
            << "\n";


    if (inputs.empty()) {

        interpreter->releaseSession(
                session
        );

        return
                result.str() +
                "FAIL: No inputs found.";
    }


    // --------------------------------------------------------
    // Print inputs
    // --------------------------------------------------------

    for (const auto& pair : inputs) {

        const std::string& name =
                pair.first;

        MNN::Tensor* tensor =
                pair.second;


        result
                << "\nInput: "
                << name
                << "\n";


        if (tensor) {

            const std::vector<int> shape =
                    tensor->shape();


            result
                    << "Shape: "
                    << shapeString(shape)
                    << "\n";


            result
                    << "Elements: "
                    << tensor->elementSize()
                    << "\n";
        }
    }


    // --------------------------------------------------------
    // Prepare inputs
    // --------------------------------------------------------

    for (const auto& pair : inputs) {

        if (
                !prepareInput(
                        interpreter.get(),
                        session,
                        pair.second,
                        pair.first
                )
        ) {

            interpreter->releaseSession(
                    session
            );

            return
                    result.str() +
                    "\nFAIL: Input preparation failed.";
        }
    }


    // --------------------------------------------------------
    // Run inference
    // --------------------------------------------------------

    result
            << "\nRunning inference...\n";


    const auto start =
            std::chrono::high_resolution_clock::now();


    MNN::ErrorCode errorCode =
            interpreter->runSession(
                    session
            );


    const auto end =
            std::chrono::high_resolution_clock::now();


    const double elapsedMs =
            std::chrono::duration<double, std::milli>(
                    end - start
            ).count();


    result
            << "Time: "
            << elapsedMs
            << " ms\n";


    result
            << "Error code: "
            << static_cast<int>(
                    errorCode
            )
            << "\n";


    if (errorCode != MNN::NO_ERROR) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: MNN inference error";

        return result.str();
    }


    // --------------------------------------------------------
    // Outputs
    // --------------------------------------------------------

    std::map<std::string, MNN::Tensor*> outputs =
            interpreter->getSessionOutputAll(
                    session
            );


    result
            << "\nOutputs: "
            << outputs.size()
            << "\n";


    for (const auto& pair : outputs) {

        const std::string& name =
                pair.first;

        MNN::Tensor* tensor =
                pair.second;


        result
                << "\nOutput: "
                << name
                << "\n";


        if (tensor) {

            const std::vector<int> shape =
                    tensor->shape();


            result
                    << "Shape: "
                    << shapeString(shape)
                    << "\n";


            result
                    << "Elements: "
                    << tensor->elementSize()
                    << "\n";
        }
    }


    // --------------------------------------------------------
    // Release
    // --------------------------------------------------------

    interpreter->releaseSession(
            session
    );


    result
            << "\nPASS: "
            << modelName
            << " executed successfully.";

    return result.str();
}


// ============================================================
// VAE-ONLY DIAGNOSTIC
//
// IMPORTANT:
// This is deliberately separate from the generic Transformer
// test.
//
// Explicit sequence:
//
// 1. Create interpreter
// 2. Create OpenCL/FP16 session
// 3. Get latent input
// 4. Explicitly resize latent to [1,32,16,16]
// 5. Explicitly resize session
// 6. Create HOST tensor
// 7. Fill deterministic latent data
// 8. copyFromHostTensor()
// 9. runSession()
// 10. copy output back to HOST
//
// MNN documents resizeTensor + resizeSession before execution
// when tensor dimensions are specified/changed, and recommends
// copyFromHostTensor for device backends.
// ============================================================

static std::string testVaeOnly(
        const std::string& vaePath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    (void) cachePath;

    std::ostringstream result;


    result
            << "VAE Decoder\n"
            << "===\n";


    // --------------------------------------------------------
    // Check file
    // --------------------------------------------------------

    struct stat fileInfo {};

    if (
            stat(
                    vaePath.c_str(),
                    &fileInfo
            ) != 0
    ) {

        result
                << "FAIL: VAE model file does not exist.\n"
                << vaePath;

        return result.str();
    }


    result
            << "Model file:\n"
            << vaePath
            << "\n\n";


    result
            << "File size: "
            << static_cast<long long>(
                    fileInfo.st_size
            )
            << " bytes\n\n";


    // --------------------------------------------------------
    // Create interpreter
    // --------------------------------------------------------

    std::shared_ptr<MNN::Interpreter>
            interpreter(
                    MNN::Interpreter::createFromFile(
                            vaePath.c_str()
                    )
            );


    if (!interpreter) {

        result
                << "FAIL: Could not create MNN VAE interpreter.";

        return result.str();
    }


    result
            << "Interpreter created.\n\n";


    // --------------------------------------------------------
    // Backend
    // --------------------------------------------------------

    MNN::ScheduleConfig config{};


    if (preferOpenCl) {

        config.type =
                MNN_FORWARD_OPENCL;

        config.numThread =
                4;

        config.mode =
                MNN_GPU_TUNING_FAST;

    } else {

        config.type =
                MNN_FORWARD_CPU;

        config.numThread =
                4;

        config.mode =
                MNN_GPU_TUNING_NONE;
    }


    MNN::BackendConfig backendConfig{};

    backendConfig.precision =
            MNN::BackendConfig::Precision_Low;

    config.backendConfig =
            &backendConfig;


    MNN::Session* session =
            interpreter->createSession(
                    config
            );


    if (!session) {

        result
                << "FAIL: Could not create VAE MNN session.";

        return result.str();
    }


    if (preferOpenCl) {

        result
                << "Backend: OpenCL / FP16\n\n";

    } else {

        result
                << "Backend: CPU\n\n";
    }


    // --------------------------------------------------------
    // Get latent input BEFORE resizing
    // --------------------------------------------------------

    MNN::Tensor* latentInput =
            interpreter->getSessionInput(
                    session,
                    "latent"
            );


    if (!latentInput) {

        // Fallback for models where the input name is not
        // exactly "latent".
        std::map<std::string, MNN::Tensor*> inputs =
                interpreter->getSessionInputAll(
                        session
                );

        if (!inputs.empty()) {

            latentInput =
                    inputs.begin()->second;
        }
    }


    if (!latentInput) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Could not find VAE latent input.";

        return result.str();
    }


    result
            << "Input: latent\n";


    result
            << "Original shape: "
            << shapeString(
                    latentInput->shape()
            )
            << "\n";


    result
            << "Original elements: "
            << latentInput->elementSize()
            << "\n\n";


    // --------------------------------------------------------
    // Explicit Sana latent shape
    //
    // Sana 0.6B 512x512:
    //
    // latent = [1,32,16,16]
    //
    // This is exactly 8192 elements.
    // --------------------------------------------------------

    const std::vector<int> latentShape = {
            1,
            32,
            16,
            16
    };


    LOGI(
            "VAE resizing latent input to [1,32,16,16]"
    );


    interpreter->resizeTensor(
            latentInput,
            latentShape
    );


    // --------------------------------------------------------
    // CRITICAL:
    //
    // resizeSession MUST happen after resizeTensor and before
    // allocating/copying the host tensor.
    // --------------------------------------------------------

    interpreter->resizeSession(
            session
    );


    // --------------------------------------------------------
    // Re-fetch input after resizeSession.
    //
    // MNN may update/recreate internal tensor resources during
    // resizeSession.
    // --------------------------------------------------------

    latentInput =
            interpreter->getSessionInput(
                    session,
                    "latent"
            );


    if (!latentInput) {

        std::map<std::string, MNN::Tensor*> inputs =
                interpreter->getSessionInputAll(
                        session
                );

        if (!inputs.empty()) {

            latentInput =
                    inputs.begin()->second;
        }
    }


    if (!latentInput) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Latent input disappeared after resizeSession.";

        return result.str();
    }


    const std::vector<int> resizedShape =
            latentInput->shape();


    result
            << "Resized shape: "
            << shapeString(resizedShape)
            << "\n";


    result
            << "Resized elements: "
            << latentInput->elementSize()
            << "\n";


    if (
            resizedShape.size() != 4 ||
            resizedShape[0] != 1 ||
            resizedShape[1] != 32 ||
            resizedShape[2] != 16 ||
            resizedShape[3] != 16
    ) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: VAE input shape is not [1,32,16,16].";

        return result.str();
    }


    if (
            latentInput->elementSize() != 8192
    ) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: VAE input element count is not 8192.";

        return result.str();
    }


    // --------------------------------------------------------
    // Create explicit HOST tensor.
    //
    // CAFFE = NCHW layout, which matches the exported ONNX
    // Sana latent tensor.
    // --------------------------------------------------------

    MNN::Tensor hostLatent(
            latentInput,
            MNN::Tensor::CAFFE
    );


    const size_t latentElements =
            hostLatent.elementSize();


    result
            << "Host latent elements: "
            << latentElements
            << "\n";


    if (
            latentElements != 8192
    ) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Host latent allocation has incorrect size.";

        return result.str();
    }


    float* latentData =
            hostLatent.host<float>();


    if (!latentData) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Host latent buffer allocation failed.";

        return result.str();
    }


    // --------------------------------------------------------
    // Deterministic diagnostic latent.
    //
    // We intentionally avoid random data here.
    //
    // A small structured signal is easier to reproduce and
    // debug across devices.
    // --------------------------------------------------------

    for (
            size_t i = 0;
            i < latentElements;
            ++i
    ) {

        const float normalized =
                static_cast<float>(
                        static_cast<int>(i % 257) - 128
                ) / 128.0f;

        latentData[i] =
                normalized * 0.05f;
    }


    // --------------------------------------------------------
    // Show a few values.
    // --------------------------------------------------------

    result
            << "Latent sample: "
            << latentData[0]
            << ", "
            << latentData[1]
            << ", "
            << latentData[2]
            << ", "
            << latentData[3]
            << "\n";


    // --------------------------------------------------------
    // Copy HOST -> DEVICE.
    //
    // This is the recommended MNN path for OpenCL/device
    // tensors.
    // --------------------------------------------------------

    const bool copyResult =
            latentInput->copyFromHostTensor(
                    &hostLatent
            );


    result
            << "Host → device copy: "
            << (
                    copyResult
                    ? "OK"
                    : "FAILED"
            )
            << "\n";


    if (!copyResult) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Unable to copy latent tensor to OpenCL device.";

        return result.str();
    }


    // --------------------------------------------------------
    // Run VAE
    // --------------------------------------------------------

    result
            << "\nRunning inference...\n";


    const auto start =
            std::chrono::high_resolution_clock::now();


    const MNN::ErrorCode errorCode =
            interpreter->runSession(
                    session
            );


    const auto end =
            std::chrono::high_resolution_clock::now();


    const double elapsedMs =
            std::chrono::duration<double, std::milli>(
                    end - start
            ).count();


    result
            << "Time: "
            << elapsedMs
            << " ms\n";


    result
            << "Error code: "
            << static_cast<int>(
                    errorCode
            )
            << "\n";


    if (
            errorCode != MNN::NO_ERROR
    ) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: MNN VAE inference error";

        return result.str();
    }


    // --------------------------------------------------------
    // Get outputs
    // --------------------------------------------------------

    std::map<std::string, MNN::Tensor*> outputs =
            interpreter->getSessionOutputAll(
                    session
            );


    result
            << "\nOutputs: "
            << outputs.size()
            << "\n";


    if (outputs.empty()) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: VAE produced no outputs.";

        return result.str();
    }


    for (const auto& pair : outputs) {

        const std::string& name =
                pair.first;

        MNN::Tensor* output =
                pair.second;


        result
                << "\nOutput: "
                << name
                << "\n";


        if (!output) {

            result
                    << "Tensor: null\n";

            continue;
        }


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


        // ----------------------------------------------------
        // Copy device output to host.
        //
        // This is intentionally done only after successful
        // inference.
        // ----------------------------------------------------

        MNN::Tensor hostOutput(
                output,
                MNN::Tensor::CAFFE
        );


        const bool outputCopy =
                output->copyToHostTensor(
                        &hostOutput
                );


        result
                << "Device → host copy: "
                << (
                        outputCopy
                        ? "OK"
                        : "FAILED"
                )
                << "\n";


        if (outputCopy) {

            float* outputData =
                    hostOutput.host<float>();


            if (
                    outputData != nullptr &&
                    hostOutput.elementSize() > 0
            ) {

                result
                        << "Output sample: "
                        << outputData[0]
                        << "\n";
            }
        }
    }


    // --------------------------------------------------------
    // Release
    // --------------------------------------------------------

    interpreter->releaseSession(
            session
    );


    result
            << "\nPASS: VAE Decoder executed successfully.";

    return result.str();
}


// ============================================================
// Transformer-only diagnostic
//
// IMPORTANT:
// Loads ONLY the Transformer.
// VAE is completely excluded.
// ============================================================

static std::string testTransformerOnly(
        const std::string& transformerPath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    return testSingleModel(
            "Transformer",
            transformerPath,
            cachePath,
            preferOpenCl
    );
}


// ============================================================
// Combined Transformer + VAE diagnostic
//
// Kept for compatibility with the existing Kotlin API.
// ============================================================

static std::string testModels(
        const std::string& transformerPath,
        const std::string& vaePath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    std::ostringstream result;


    result
            << "SANA MODEL TEST\n"
            << "================\n\n";


    // --------------------------------------------------------
    // Transformer
    // --------------------------------------------------------

    result
            << "TRANSFORMER TEST\n"
            << "----------------\n";


    result
            << testSingleModel(
                    "Transformer",
                    transformerPath,
                    cachePath,
                    preferOpenCl
            );


    result
            << "\n\n";


    // --------------------------------------------------------
    // VAE
    // --------------------------------------------------------

    result
            << "VAE TEST\n"
            << "--------\n";


    result
            << testVaeOnly(
                    vaePath,
                    cachePath,
                    preferOpenCl
            );


    return result.str();
}


// ============================================================
// JNI: nativeInitialize
// ============================================================

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

    (void) assetManager;


    try {

        if (
                modelAsset == nullptr
        ) {

            return JNI_FALSE;
        }


        const std::string model =
                jstringToString(
                        env,
                        modelAsset
                );


        const std::string cache =
                jstringToString(
                        env,
                        cachePath
                );


        if (model.empty()) {

            return JNI_FALSE;
        }


        const int threads =
                std::max(
                        1,
                        std::min(
                                8,
                                static_cast<int>(
                                        cpuThreads
                                )
                        )
                );


        const bool initialized =
                gEngine.initialize(
                        model,
                        cache,
                        preferOpenCl == JNI_TRUE,
                        threads
                );


        return initialized
                ? JNI_TRUE
                : JNI_FALSE;

    } catch (
            const std::exception& e
    ) {

        LOGE(
                "nativeInitialize exception: %s",
                e.what()
        );

        return JNI_FALSE;

    } catch (...) {

        LOGE(
                "nativeInitialize unknown exception"
        );

        return JNI_FALSE;
    }
}


// ============================================================
// JNI: nativeIsInitialized
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv*,
        jobject
) {

    try {

        return gEngine.isInitialized()
                ? JNI_TRUE
                : JNI_FALSE;

    } catch (...) {

        return JNI_FALSE;
    }
}


// ============================================================
// JNI: nativeGetBackend
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject
) {

    try {

        const std::string value =
                gEngine.backend();

        return env->NewStringUTF(
                value.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown"
        );
    }
}


// ============================================================
// JNI: nativeGetStatus
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject
) {

    try {

        const std::string value =
                gEngine.status();

        return env->NewStringUTF(
                value.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown"
        );
    }
}


// ============================================================
// JNI: nativeRelease
// ============================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject
) {

    try {

        gEngine.release();

    } catch (...) {

        LOGE(
                "nativeRelease exception"
        );
    }
}


// ============================================================
// JNI: nativeTestTransformer
// ============================================================

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

        if (
                transformerPath == nullptr
        ) {

            return env->NewStringUTF(
                    "FAIL: Transformer path is null."
            );
        }


        const std::string transformer =
                jstringToString(
                        env,
                        transformerPath
                );


        const std::string cache =
                jstringToString(
                        env,
                        cachePath
                );


        const std::string output =
                testTransformerOnly(
                        transformer,
                        cache,
                        preferOpenCl == JNI_TRUE
                );


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (
            const std::exception& e
    ) {

        LOGE(
                "nativeTestTransformer exception: %s",
                e.what()
        );


        const std::string output =
                std::string(
                        "FAIL: Native Transformer exception: "
                ) +
                e.what();


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "FAIL: Unknown native Transformer exception."
        );
    }
}


// ============================================================
// JNI: nativeTestVae
//
// This is the NEW corrected VAE diagnostic.
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl
) {

    try {

        if (
                vaePath == nullptr
        ) {

            return env->NewStringUTF(
                    "FAIL: VAE path is null."
            );
        }


        const std::string vae =
                jstringToString(
                        env,
                        vaePath
                );


        const std::string cache =
                jstringToString(
                        env,
                        cachePath
                );


        const std::string output =
                testVaeOnly(
                        vae,
                        cache,
                        preferOpenCl == JNI_TRUE
                );


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (
            const std::exception& e
    ) {

        LOGE(
                "nativeTestVae exception: %s",
                e.what()
        );


        const std::string output =
                std::string(
                        "FAIL: Native VAE exception: "
                ) +
                e.what();


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "FAIL: Unknown native VAE exception."
        );
    }
}


// ============================================================
// JNI: nativeTestModels
//
// Legacy combined diagnostic kept for compatibility.
// ============================================================

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
                    "FAIL: Transformer or VAE path is null."
            );
        }


        const std::string transformer =
                jstringToString(
                        env,
                        transformerPath
                );


        const std::string vae =
                jstringToString(
                        env,
                        vaePath
                );


        const std::string cache =
                jstringToString(
                        env,
                        cachePath
                );


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

    } catch (
            const std::exception& e
    ) {

        LOGE(
                "nativeTestModels exception: %s",
                e.what()
        );


        const std::string output =
                std::string(
                        "FAIL: Native model test exception: "
                ) +
                e.what();


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "FAIL: Unknown native model test exception."
        );
    }
}
