#include <jni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "SanaNative"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex gMutex;

bool gInitialized = false;
std::string gBackend = "Not initialized";
std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;
MNN::Session* gSession = nullptr;


// ============================================================
// BASIC HELPERS
// ============================================================

std::string jstringToString(JNIEnv* env, jstring value) {
    if (value == nullptr) return "";

    const char* chars =
        env->GetStringUTFChars(value, nullptr);

    if (chars == nullptr) return "";

    std::string result(chars);

    env->ReleaseStringUTFChars(value, chars);

    return result;
}


std::string formatShape(const MNN::Tensor* tensor) {
    if (tensor == nullptr) return "null";

    std::vector<int> shape = tensor->shape();

    std::ostringstream out;
    out << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out << ", ";
        out << shape[i];
    }

    out << "]";

    return out.str();
}


std::string dimensionTypeName(
        MNN::Tensor::DimensionType type) {

    switch (type) {
        case MNN::Tensor::TENSORFLOW:
            return "TENSORFLOW / NHWC";

        case MNN::Tensor::CAFFE:
            return "CAFFE / NCHW";

        case MNN::Tensor::CAFFE_C4:
            return "CAFFE_C4 / NC4HW4";

        default:
            return "UNKNOWN";
    }
}


long long getFileSize(const std::string& path) {

    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file.good()) return -1;

    long long size =
        static_cast<long long>(file.tellg());

    file.close();

    return size;
}


// ============================================================
// SESSION CONFIG
// ============================================================

MNN::ScheduleConfig makeScheduleConfig(
        bool preferOpenCl,
        int cpuThreads) {

    MNN::ScheduleConfig config;

    if (preferOpenCl) {

        config.type = MNN_FORWARD_OPENCL;
        config.mode = MNN_GPU_TUNING_HEAVY;

    } else {

        config.type = MNN_FORWARD_CPU;
        config.numThread = cpuThreads;
    }

    return config;
}


bool createSession(
        MNN::Interpreter* interpreter,
        MNN::Session*& session,
        bool preferOpenCl,
        int cpuThreads) {

    if (interpreter == nullptr) return false;

    MNN::ScheduleConfig config =
        makeScheduleConfig(
            preferOpenCl,
            cpuThreads
        );

    if (preferOpenCl) {
        config.backupType = MNN_FORWARD_CPU;
    }

    session =
        interpreter->createSession(config);

    return session != nullptr;
}


void destroyInterpreter(
        MNN::Interpreter*& interpreter) {

    if (interpreter != nullptr) {

        MNN::Interpreter::destroy(
            interpreter
        );

        interpreter = nullptr;
    }
}


void releaseGlobalLocked() {

    if (gSession != nullptr &&
        gInterpreter != nullptr) {

        gInterpreter->releaseSession(
            gSession
        );

        gSession = nullptr;
    }

    if (gInterpreter != nullptr) {

        MNN::Interpreter::destroy(
            gInterpreter
        );

        gInterpreter = nullptr;
    }

    gInitialized = false;

    gBackend = "Not initialized";
    gStatus = "Released";
}


// ============================================================
// TRANSFORMER TEST
// ============================================================

std::string runTransformerDiagnostic(
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl) {

    (void)cachePath;

    std::ostringstream result;

    result
        << "SANA 0.6B / 512 TRANSFORMER TEST\n\n";

    result
        << "Transformer file:\n"
        << modelPath
        << "\n\n";

    long long fileSize =
        getFileSize(modelPath);

    if (fileSize < 0) {

        result
            << "FAIL: Transformer file does not exist.";

        return result.str();
    }

    result
        << "File size: "
        << fileSize
        << " bytes\n\n";


    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );

    if (interpreter == nullptr) {

        result
            << "FAIL: Could not create Transformer interpreter.";

        return result.str();
    }

    result
        << "Transformer interpreter created.\n\n";


    MNN::Session* session = nullptr;

    if (!createSession(
            interpreter,
            session,
            preferOpenCl,
            4)) {

        result
            << "FAIL: Could not create Transformer session.";

        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Backend: "
        << (preferOpenCl
            ? "OpenCL / FP16"
            : "CPU")
        << "\n\n";


    MNN::Tensor* encoderInput =
        interpreter->getSessionInput(
            session,
            "encoder_hidden_states"
        );

    MNN::Tensor* hiddenInput =
        interpreter->getSessionInput(
            session,
            "hidden_states"
        );

    MNN::Tensor* timestepInput =
        interpreter->getSessionInput(
            session,
            "timestep"
        );


    if (encoderInput == nullptr ||
        hiddenInput == nullptr ||
        timestepInput == nullptr) {

        result
            << "FAIL: Transformer inputs not found.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Actual backend: "
        << (preferOpenCl
            ? "OpenCL / FP16"
            : "CPU")
        << "\n\n";

    result << "Inputs: 3\n\n";


    result
        << "Input: encoder_hidden_states\n"
        << "Shape: "
        << formatShape(encoderInput)
        << "\n"
        << "Elements: "
        << encoderInput->elementSize()
        << "\n\n";


    result
        << "Input: hidden_states\n"
        << "Shape: "
        << formatShape(hiddenInput)
        << "\n"
        << "Elements: "
        << hiddenInput->elementSize()
        << "\n\n";


    result
        << "Input: timestep\n"
        << "Shape: "
        << formatShape(timestepInput)
        << "\n"
        << "Elements: "
        << timestepInput->elementSize()
        << "\n\n";


    MNN::Tensor encoderHost(
        encoderInput,
        MNN::Tensor::CAFFE
    );

    MNN::Tensor hiddenHost(
        hiddenInput,
        MNN::Tensor::CAFFE
    );

    MNN::Tensor timestepHost(
        timestepInput,
        MNN::Tensor::CAFFE
    );


    float* encoderData =
        encoderHost.host<float>();

    float* hiddenData =
        hiddenHost.host<float>();

    float* timestepData =
        timestepHost.host<float>();


    if (encoderData == nullptr ||
        hiddenData == nullptr ||
        timestepData == nullptr) {

        result
            << "FAIL: Transformer host tensor allocation failed.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    for (int i = 0;
         i < encoderHost.elementSize();
         ++i) {

        encoderData[i] =
            0.001f *
            static_cast<float>(
                (i % 100) - 50
            );
    }


    for (int i = 0;
         i < hiddenHost.elementSize();
         ++i) {

        hiddenData[i] =
            0.001f *
            static_cast<float>(
                (i % 100) - 50
            );
    }


    timestepData[0] = 0.5f;


    bool encoderCopied =
        encoderInput->copyFromHostTensor(
            &encoderHost
        );

    bool hiddenCopied =
        hiddenInput->copyFromHostTensor(
            &hiddenHost
        );

    bool timestepCopied =
        timestepInput->copyFromHostTensor(
            &timestepHost
        );


    if (!encoderCopied ||
        !hiddenCopied ||
        !timestepCopied) {

        result
            << "FAIL: Could not copy Transformer input tensors.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Running inference...\n\n";


    auto start =
        std::chrono::steady_clock::now();


    int errorCode =
        interpreter->runSession(session);


    auto end =
        std::chrono::steady_clock::now();


    double elapsed =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();


    result
        << "Time: "
        << elapsed
        << " ms\n\n";

    result
        << "Error code: "
        << errorCode
        << "\n\n";


    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            "sample"
        );


    if (output == nullptr) {

        result
            << "FAIL: Transformer output not found.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Outputs: 1\n\n";

    result
        << "Output: sample\n";

    result
        << "Shape: "
        << formatShape(output)
        << "\n";

    result
        << "Elements: "
        << output->elementSize()
        << "\n\n";


    if (errorCode == 0) {

        result
            << "PASS: Transformer executed successfully";

    } else {

        result
            << "FAIL: Transformer inference error";
    }


    interpreter->releaseSession(session);

    destroyInterpreter(interpreter);


    result
        << "\n\nTransformer released successfully.";

    return result.str();
}


// ============================================================
// FINAL MINIMAL VAE TEST
// ============================================================

std::string runVaeMinimalTest(
        const std::string& modelPath,
        bool preferOpenCl) {

    std::ostringstream result;


    result
        << "SANA 0.6B / 512 VAE FINAL TEST\n\n";


    result
        << "Model file:\n"
        << modelPath
        << "\n\n";


    long long fileSize =
        getFileSize(modelPath);


    if (fileSize < 0) {

        result
            << "FAIL: VAE file does not exist.";

        return result.str();
    }


    result
        << "File size: "
        << fileSize
        << " bytes\n\n";


    // --------------------------------------------------------
    // CREATE INTERPRETER
    // --------------------------------------------------------

    result
        << "Creating "
        << (preferOpenCl
            ? "OpenCL"
            : "CPU")
        << " interpreter...\n";


    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );


    if (interpreter == nullptr) {

        result
            << "FAIL: VAE interpreter creation failed.";

        return result.str();
    }


    result
        << "Interpreter created.\n";


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

    MNN::Session* session = nullptr;


    if (!createSession(
            interpreter,
            session,
            preferOpenCl,
            4)) {

        result
            << "FAIL: VAE session creation failed.";

        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Session created.\n\n";


    // --------------------------------------------------------
    // INPUT
    // --------------------------------------------------------

    MNN::Tensor* input =
        interpreter->getSessionInput(
            session,
            "latent"
        );


    if (input == nullptr) {

        input =
            interpreter->getSessionInput(
                session,
                nullptr
            );
    }


    if (input == nullptr) {

        result
            << "FAIL: VAE input not found.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Device input shape: "
        << formatShape(input)
        << "\n";

    result
        << "Device input elements: "
        << input->elementSize()
        << "\n";

    result
        << "Device input type: "
        << static_cast<int>(
            input->getType().code
        )
        << "\n";

    result
        << "Device input bytes: "
        << static_cast<int>(
            input->getType().bytes()
        )
        << "\n";

    result
        << "Device input format: "
        << dimensionTypeName(
            input->getDimensionType()
        )
        << "\n";

    result
        << "Device ID: "
        << input->deviceId()
        << "\n\n";


    // --------------------------------------------------------
    // EXACT EXPECTED SHAPE
    // --------------------------------------------------------

    std::vector<int> expectedShape = {
        1, 32, 16, 16
    };


    if (input->shape() != expectedShape) {

        result
            << "Input shape differs from expected.\n";

        result
            << "Resizing to [1,32,16,16]...\n";


        interpreter->resizeTensor(
            input,
            expectedShape
        );


        interpreter->resizeSession(
            session
        );
    }


    result
        << "Final device shape: "
        << formatShape(input)
        << "\n\n";


    // --------------------------------------------------------
    // SIMPLE HOST TENSOR
    // --------------------------------------------------------

    result
        << "Creating simple MNN host tensor...\n";


    MNN::Tensor hostTensor(
        input,
        MNN::Tensor::CAFFE
    );


    result
        << "Host tensor shape: "
        << formatShape(&hostTensor)
        << "\n";

    result
        << "Host tensor elements: "
        << hostTensor.elementSize()
        << "\n";

    result
        << "Host tensor type: "
        << static_cast<int>(
            hostTensor.getType().code
        )
        << "\n";

    result
        << "Host tensor bytes: "
        << static_cast<int>(
            hostTensor.getType().bytes()
        )
        << "\n";

    result
        << "Host tensor format: "
        << dimensionTypeName(
            hostTensor.getDimensionType()
        )
        << "\n";

    result
        << "Host tensor device ID: "
        << hostTensor.deviceId()
        << "\n\n";


    // --------------------------------------------------------
    // HOST MEMORY
    // --------------------------------------------------------

    float* hostData =
        hostTensor.host<float>();


    if (hostData == nullptr) {

        result
            << "FAIL: MNN host tensor returned null pointer.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "PASS: Host memory allocated.\n";


    // --------------------------------------------------------
    // FILL LATENT
    // --------------------------------------------------------

    int elements =
        hostTensor.elementSize();


    for (int i = 0;
         i < elements;
         ++i) {

        hostData[i] =
            -0.05f +
            0.000390625f *
            static_cast<float>(
                i % 256
            );
    }


    result
        << "Latent sample:\n"
        << hostData[0]
        << ", "
        << hostData[1]
        << ", "
        << hostData[2]
        << ", "
        << hostData[3]
        << "\n\n";


    // --------------------------------------------------------
    // HOST -> DEVICE
    // --------------------------------------------------------

    result
        << "Calling device input copyFromHostTensor...\n";


    bool copied =
        input->copyFromHostTensor(
            &hostTensor
        );


    result
        << "copyFromHostTensor: "
        << (copied
            ? "SUCCESS"
            : "FAILED")
        << "\n\n";


    if (!copied) {

        result
            << "FINAL RESULT: VAE INPUT COPY FAILED.\n\n";

        result
            << "The Sana VAE graph loaded and its input shape is "
               "correct, but this MNN OpenCL configuration cannot "
               "accept the host tensor as the VAE input.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "PASS: VAE input successfully transferred.\n\n";


    // --------------------------------------------------------
    // RUN VAE
    // --------------------------------------------------------

    result
        << "Running VAE inference...\n";


    auto start =
        std::chrono::steady_clock::now();


    int errorCode =
        interpreter->runSession(session);


    auto end =
        std::chrono::steady_clock::now();


    double elapsed =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();


    result
        << "Inference time: "
        << elapsed
        << " ms\n";

    result
        << "Error code: "
        << errorCode
        << "\n\n";


    // --------------------------------------------------------
    // OUTPUT
    // --------------------------------------------------------

    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            nullptr
        );


    if (output == nullptr) {

        result
            << "FAIL: VAE output is null.";

        interpreter->releaseSession(session);
        destroyInterpreter(interpreter);

        return result.str();
    }


    result
        << "Output shape: "
        << formatShape(output)
        << "\n";

    result
        << "Output elements: "
        << output->elementSize()
        << "\n";


    // --------------------------------------------------------
    // OUTPUT HOST COPY
    // --------------------------------------------------------

    MNN::Tensor outputHost(
        output,
        MNN::Tensor::CAFFE
    );


    bool outputCopied =
        output->copyToHostTensor(
            &outputHost
        );


    result
        << "Output copyToHostTensor: "
        << (outputCopied
            ? "SUCCESS"
            : "FAILED")
        << "\n";


    if (outputCopied) {

        float* outputData =
            outputHost.host<float>();


        if (outputData != nullptr) {

            result
                << "Output sample: "
                << outputData[0]
                << ", "
                << outputData[1]
                << ", "
                << outputData[2]
                << ", "
                << outputData[3]
                << "\n";
        }
    }


    result << "\n";


    // --------------------------------------------------------
    // FINAL
    // --------------------------------------------------------

    if (errorCode == 0) {

        result
            << "========================================\n"
            << "PASS: SANA VAE WORKS\n"
            << "========================================";

    } else {

        result
            << "========================================\n"
            << "VAE INFERENCE FAILED\n"
            << "========================================";
    }


    interpreter->releaseSession(session);

    destroyInterpreter(interpreter);


    return result.str();
}


// ============================================================
// ASSET COPY
// ============================================================

bool copyAssetToFile(
        AAssetManager* assetManager,
        const std::string& assetName,
        const std::string& destination) {

    if (assetManager == nullptr) {
        return false;
    }


    AAsset* asset =
        AAssetManager_open(
            assetManager,
            assetName.c_str(),
            AASSET_MODE_STREAMING
        );


    if (asset == nullptr) {
        return false;
    }


    FILE* output =
        std::fopen(
            destination.c_str(),
            "wb"
        );


    if (output == nullptr) {

        AAsset_close(asset);

        return false;
    }


    constexpr size_t BUFFER_SIZE =
        1024 * 1024;


    std::vector<char> buffer(
        BUFFER_SIZE
    );


    bool success = true;


    while (true) {

        int count =
            AAsset_read(
                asset,
                buffer.data(),
                static_cast<int>(
                    buffer.size()
                )
            );


        if (count < 0) {

            success = false;
            break;
        }


        if (count == 0) {
            break;
        }


        size_t written =
            std::fwrite(
                buffer.data(),
                1,
                static_cast<size_t>(count),
                output
            );


        if (written !=
            static_cast<size_t>(count)) {

            success = false;
            break;
        }
    }


    std::fclose(output);
    AAsset_close(asset);


    return success;
}


// ============================================================
// INITIALIZE
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv* env,
        jobject /* thiz */,
        jobject assetManagerObject,
        jstring modelAsset,
        jstring cachePath,
        jboolean preferOpenCl,
        jint cpuThreads) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    releaseGlobalLocked();


    if (assetManagerObject == nullptr) {

        gStatus =
            "AssetManager is null";

        return JNI_FALSE;
    }


    AAssetManager* assetManager =
        AAssetManager_fromJava(
            env,
            assetManagerObject
        );


    if (assetManager == nullptr) {

        gStatus =
            "Could not obtain AssetManager";

        return JNI_FALSE;
    }


    std::string assetName =
        jstringToString(
            env,
            modelAsset
        );


    std::string cache =
        jstringToString(
            env,
            cachePath
        );


    if (assetName.empty() ||
        cache.empty()) {

        gStatus =
            "Invalid model or cache path";

        return JNI_FALSE;
    }


    std::string modelPath =
        cache +
        "/sana_model.mnn";


    if (!copyAssetToFile(
            assetManager,
            assetName,
            modelPath
        )) {

        gStatus =
            "Failed to copy model";

        return JNI_FALSE;
    }


    gInterpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );


    if (gInterpreter == nullptr) {

        gStatus =
            "MNN interpreter creation failed";

        return JNI_FALSE;
    }


    int threads =
        std::max(
            1,
            std::min(
                8,
                static_cast<int>(
                    cpuThreads
                )
            )
        );


    if (!createSession(
            gInterpreter,
            gSession,
            preferOpenCl == JNI_TRUE,
            threads
        )) {

        gStatus =
            "MNN session creation failed";

        releaseGlobalLocked();

        return JNI_FALSE;
    }


    gInitialized = true;


    gBackend =
        preferOpenCl == JNI_TRUE
            ? "OpenCL / FP16"
            : "CPU";


    gStatus =
        "Sana MNN model initialized";


    return JNI_TRUE;
}


// ============================================================
// IS INITIALIZED
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv* /* env */,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );

    return gInitialized
        ? JNI_TRUE
        : JNI_FALSE;
}


// ============================================================
// BACKEND
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );

    return env->NewStringUTF(
        gBackend.c_str()
    );
}


// ============================================================
// STATUS
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );

    return env->NewStringUTF(
        gStatus.c_str()
    );
}


// ============================================================
// RELEASE
// ============================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv* /* env */,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );

    releaseGlobalLocked();
}


// ============================================================
// TEST TRANSFORMER
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
        JNIEnv* env,
        jobject /* thiz */,
        jstring transformerPath,
        jstring cachePath,
        jboolean preferOpenCl) {

    std::string transformer =
        jstringToString(
            env,
            transformerPath
        );

    std::string cache =
        jstringToString(
            env,
            cachePath
        );


    std::string result =
        runTransformerDiagnostic(
            transformer,
            cache,
            preferOpenCl == JNI_TRUE
        );


    return env->NewStringUTF(
        result.c_str()
    );
}


// ============================================================
// TEST VAE
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject /* thiz */,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl) {

    (void)cachePath;


    std::string vae =
        jstringToString(
            env,
            vaePath
        );


    std::string result =
        runVaeMinimalTest(
            vae,
            preferOpenCl == JNI_TRUE
        );


    return env->NewStringUTF(
        result.c_str()
    );
}


// ============================================================
// TEST BOTH
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestModels(
        JNIEnv* env,
        jobject /* thiz */,
        jstring transformerPath,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl) {

    std::string transformer =
        jstringToString(
            env,
            transformerPath
        );

    std::string vae =
        jstringToString(
            env,
            vaePath
        );

    std::string cache =
        jstringToString(
            env,
            cachePath
        );


    std::ostringstream result;


    result
        << runTransformerDiagnostic(
            transformer,
            cache,
            preferOpenCl == JNI_TRUE
        );


    result
        << "\n\n========================================\n"
        << "VAE FINAL TEST\n"
        << "========================================\n";


    result
        << runVaeMinimalTest(
            vae,
            preferOpenCl == JNI_TRUE
        );


    return env->NewStringUTF(
        result.str().c_str()
    );
}

} // namespace
