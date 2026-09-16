#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>


// ============================================================
// LOGGING
// ============================================================

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


// ============================================================
// UTILITIES
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
// PERSISTENT SANA ENGINE
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
// TRANSFORMER INPUT PREPARATION
//
// PRESERVED FOR THE WORKING TRANSFORMER TEST.
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
// GENERIC MODEL DIAGNOSTIC
//
// PRESERVED FOR TRANSFORMER.
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


    interpreter->resizeSession(
            session
    );


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

            result
                    << "Shape: "
                    << shapeString(
                            tensor->shape()
                    )
                    << "\n";


            result
                    << "Elements: "
                    << tensor->elementSize()
                    << "\n";
        }
    }


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


    if (
            errorCode != MNN::NO_ERROR
    ) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: MNN inference error";

        return result.str();
    }


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

            result
                    << "Shape: "
                    << shapeString(
                            tensor->shape()
                    )
                    << "\n";


            result
                    << "Elements: "
                    << tensor->elementSize()
                    << "\n";
        }
    }


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
// VAE DIAGNOSTIC
//
// NEW APPROACH:
//
// DO NOT use copyFromHostTensor() for the VAE input.
//
// Instead:
//
// resizeTensor()
// resizeSession()
// getSessionInput()
// map(MAP_TENSOR_WRITE)
// write latent directly
// unmap(...)
// runSession()
//
// This follows MNN's documented device-tensor mapping path.
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
    // File
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
    // Interpreter
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
    // OpenCL / CPU
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
    // Find latent
    // --------------------------------------------------------

    MNN::Tensor* latentInput =
            interpreter->getSessionInput(
                    session,
                    "latent"
            );


    if (!latentInput) {

        latentInput =
                interpreter->getSessionInput(
                        session,
                        nullptr
                );
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
            << "\n";


    // --------------------------------------------------------
    // Print original tensor properties
    // --------------------------------------------------------

    result
            << "Original tensor type code: "
            << latentInput->getType().code
            << "\n";


    result
            << "Original tensor bytes: "
            << latentInput->getType().bytes()
            << "\n";


    result
            << "Original device ID: "
            << static_cast<unsigned long long>(
                    latentInput->deviceId()
            )
            << "\n\n";


    // --------------------------------------------------------
    // Sana latent
    // --------------------------------------------------------

    const std::vector<int> latentShape = {
            1,
            32,
            16,
            16
    };


    LOGI(
            "VAE resizeTensor -> [1,32,16,16]"
    );


    interpreter->resizeTensor(
            latentInput,
            latentShape
    );


    // --------------------------------------------------------
    // CRITICAL
    // --------------------------------------------------------

    interpreter->resizeSession(
            session
    );


    // --------------------------------------------------------
    // Re-fetch after resize
    // --------------------------------------------------------

    latentInput =
            interpreter->getSessionInput(
                    session,
                    "latent"
            );


    if (!latentInput) {

        latentInput =
                interpreter->getSessionInput(
                        session,
                        nullptr
                );
    }


    if (!latentInput) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Latent disappeared after resizeSession.";

        return result.str();
    }


    const std::vector<int> resizedShape =
            latentInput->shape();


    result
            << "Resized shape: "
            << shapeString(
                    resizedShape
            )
            << "\n";


    result
            << "Resized elements: "
            << latentInput->elementSize()
            << "\n";


    result
            << "Resized tensor type code: "
            << latentInput->getType().code
            << "\n";


    result
            << "Resized tensor bytes: "
            << latentInput->getType().bytes()
            << "\n";


    result
            << "Resized device ID: "
            << static_cast<unsigned long long>(
                    latentInput->deviceId()
            )
            << "\n";


    // --------------------------------------------------------
    // Validate shape
    // --------------------------------------------------------

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
    // MAP DEVICE TENSOR
    // --------------------------------------------------------

    result
            << "\nMapping OpenCL input tensor for write...\n";


    void* mappedMemory =
            latentInput->map(
                    MNN::Tensor::MAP_TENSOR_WRITE,
                    latentInput->getDimensionType()
            );


    if (!mappedMemory) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: MNN could not map VAE input tensor for WRITE.";

        return result.str();
    }


    result
            << "Tensor map: OK\n";


    // --------------------------------------------------------
    // IMPORTANT
    //
    // map() gives the backend-controlled memory representation.
    //
    // For the diagnostic we assume float data because the
    // exported Sana latent input is FP32/FP16-compatible.
    // --------------------------------------------------------

    float* latentData =
            static_cast<float*>(
                    mappedMemory
            );


    if (!latentData) {

        latentInput->unmap(
                MNN::Tensor::MAP_TENSOR_WRITE,
                latentInput->getDimensionType(),
                mappedMemory
        );


        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: Mapped VAE tensor pointer is null.";

        return result.str();
    }


    // --------------------------------------------------------
    // Fill deterministic latent
    // --------------------------------------------------------

    const size_t elementCount =
            8192;


    for (
            size_t i = 0;
            i < elementCount;
            ++i
    ) {

        const float normalized =
                static_cast<float>(
                        static_cast<int>(i % 257) - 128
                ) / 128.0f;

        latentData[i] =
                normalized * 0.05f;
    }


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
    // UNMAP
    // --------------------------------------------------------

    latentInput->unmap(
            MNN::Tensor::MAP_TENSOR_WRITE,
            latentInput->getDimensionType(),
            mappedMemory
    );


    result
            << "Tensor unmap: OK\n";


    // --------------------------------------------------------
    // RUN
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


    // --------------------------------------------------------
    // Failure
    // --------------------------------------------------------

    if (
            errorCode != MNN::NO_ERROR
    ) {

        interpreter->releaseSession(
                session
        );

        result
                << "FAIL: MNN VAE inference error.";

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


        result
                << "Type code: "
                << output->getType().code
                << "\n";


        result
                << "Bytes: "
                << output->getType().bytes()
                << "\n";
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
// TRANSFORMER
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
// COMBINED TEST
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
// JNI: INITIALIZE
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
// JNI: IS INITIALIZED
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
// JNI: BACKEND
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
// JNI: STATUS
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
// JNI: RELEASE
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
// JNI: TEST TRANSFORMER
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
// JNI: TEST VAE
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
// JNI: COMBINED TEST
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
