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

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex gMutex;

bool gInitialized = false;

std::string gBackend = "Not initialized";
std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;
MNN::Session* gSession = nullptr;


// ------------------------------------------------------------
// JNI STRING HELPER
// ------------------------------------------------------------

std::string jstringToString(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return "";
    }

    const char* chars = env->GetStringUTFChars(value, nullptr);

    if (chars == nullptr) {
        return "";
    }

    std::string result(chars);

    env->ReleaseStringUTFChars(value, chars);

    return result;
}


// ------------------------------------------------------------
// SHAPE HELPER
// ------------------------------------------------------------

std::string formatShape(const MNN::Tensor* tensor) {

    if (tensor == nullptr) {
        return "null";
    }

    std::ostringstream out;

    out << "[";

    for (int i = 0; i < tensor->dimensions(); ++i) {

        if (i > 0) {
            out << ", ";

        }

        out << tensor->length(i);
    }

    out << "]";

    return out.str();
}


// ------------------------------------------------------------
// DIMENSION TYPE
// ------------------------------------------------------------

std::string dimensionTypeName(MNN::Tensor::DimensionType type) {

    switch (type) {

        case MNN::Tensor::TENSORFLOW:
            return "TENSORFLOW / NHWC";

        case MNN::CAFFE:
            return "CAFFE / NCHW";

        case MNN::CAFFE_C4:
            return "CAFFE_C4";

        default:
            return "UNKNOWN";
    }
}


// ------------------------------------------------------------
// FILE SIZE
// ------------------------------------------------------------

long long getFileSize(const std::string& path) {

    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file.good()) {
        return -1;
    }

    return static_cast<long long>(file.tellg());
}


// ------------------------------------------------------------
// SCHEDULE CONFIG
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// CREATE SESSION
// ------------------------------------------------------------

MNN::Session* createSession(
        MNN::Interpreter* interpreter,
        bool preferOpenCl) {

    if (interpreter == nullptr) {
        return nullptr;
    }

    MNN::ScheduleConfig config =
            makeScheduleConfig(
                    preferOpenCl,
                    4);

    /*
     * Keep CPU as backup for OpenCL.
     *
     * This is the same configuration used by the
     * previously working Transformer test.
     */

    if (preferOpenCl) {
        config.backupType = MNN_FORWARD_CPU;
    }

    MNN::Session* session =
            interpreter->createSession(config);

    return session;
}


// ------------------------------------------------------------
// DESTROY INTERPRETER
// ------------------------------------------------------------

void destroyInterpreter(
        MNN::Interpreter*& interpreter,
        MNN::Session*& session) {

    if (interpreter != nullptr) {

        if (session != nullptr) {

            interpreter->releaseSession(session);

            session = nullptr;
        }

        interpreter->release();

        interpreter = nullptr;
    }
}


// ------------------------------------------------------------
// RELEASE GLOBAL STATE
// ------------------------------------------------------------

void releaseGlobalLocked() {

    if (gInterpreter != nullptr) {

        if (gSession != nullptr) {

            gInterpreter->releaseSession(gSession);

            gSession = nullptr;
        }

        gInterpreter->release();

        gInterpreter = nullptr;
    }

    gInitialized = false;

    gBackend = "Not initialized";

    gStatus = "Released";
}


// ------------------------------------------------------------
// TRANSFORMER DIAGNOSTIC
// ------------------------------------------------------------

std::string runTransformerDiagnostic(
        const std::string& transformerPath,
        const std::string& cachePath,
        bool preferOpenCl) {

    std::ostringstream report;

    report << "SANA 0.6B / 512 TRANSFORMER TEST\n\n";

    report << "Transformer file:\n";
    report << transformerPath << "\n\n";

    long long fileSize =
            getFileSize(transformerPath);

    report << "File size: "
           << fileSize
           << " bytes\n\n";

    if (fileSize <= 0) {

        report << "FAIL: Transformer file not found or empty.\n";

        return report.str();
    }


    // --------------------------------------------------------
    // CREATE INTERPRETER
    // --------------------------------------------------------

    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    transformerPath.c_str());

    if (interpreter == nullptr) {

        report << "FAIL: Could not create Transformer interpreter.\n";

        return report.str();
    }

    report << "Transformer interpreter created.\n\n";


    // --------------------------------------------------------
    // CREATE SESSION
    // --------------------------------------------------------

    MNN::Session* session =
            createSession(
                    interpreter,
                    preferOpenCl);

    if (session == nullptr) {

        report << "FAIL: Could not create Transformer session.\n";

        interpreter->release();

        return report.str();
    }


    if (preferOpenCl) {

        report << "Backend: OpenCL / FP16\n";

    } else {

        report << "Backend: CPU\n";
    }


    // --------------------------------------------------------
    // GET ACTUAL BACKEND
    // --------------------------------------------------------

    report << "\nActual backend requested successfully.\n\n";


    // --------------------------------------------------------
    // INPUTS
    // --------------------------------------------------------

    MNN::Tensor* encoderInput =
            interpreter->getSessionInput(
                    session,
                    "encoder_hidden_states");

    MNN::Tensor* hiddenInput =
            interpreter->getSessionInput(
                    session,
                    "hidden_states");

    MNN::Tensor* timestepInput =
            interpreter->getSessionInput(
                    session,
                    "timestep");


    if (encoderInput == nullptr ||
        hiddenInput == nullptr ||
        timestepInput == nullptr) {

        report << "FAIL: Required Transformer inputs not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // LOG INPUTS
    // --------------------------------------------------------

    report << "Inputs: 3\n\n";

    report << "Input: encoder_hidden_states\n";
    report << "Shape: "
           << formatShape(encoderInput)
           << "\n";

    report << "Elements: "
           << encoderInput->elementSize()
           << "\n\n";


    report << "Input: hidden_states\n";
    report << "Shape: "
           << formatShape(hiddenInput)
           << "\n";

    report << "Elements: "
           << hiddenInput->elementSize()
           << "\n\n";


    report << "Input: timestep\n";
    report << "Shape: "
           << formatShape(timestepInput)
           << "\n";

    report << "Elements: "
           << timestepInput->elementSize()
           << "\n\n";


    // --------------------------------------------------------
    // HOST TENSORS
    // --------------------------------------------------------

    MNN::Tensor encoderHost(
            encoderInput,
            MNN::Tensor::CAFFE);

    MNN::Tensor hiddenHost(
            hiddenInput,
            MNN::Tensor::CAFFE);

    MNN::Tensor timestepHost(
            timestepInput,
            MNN::Tensor::CAFFE);


    // --------------------------------------------------------
    // FILL ENCODER INPUT
    // --------------------------------------------------------

    float* encoderData =
            encoderHost.host<float>();

    if (encoderData == nullptr) {

        report << "FAIL: Encoder host tensor allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }

    std::fill(
            encoderData,
            encoderData + encoderHost.elementSize(),
            0.0f);


    // --------------------------------------------------------
    // FILL HIDDEN INPUT
    // --------------------------------------------------------

    float* hiddenData =
            hiddenHost.host<float>();

    if (hiddenData == nullptr) {

        report << "FAIL: Hidden host tensor allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }

    std::fill(
            hiddenData,
            hiddenData + hiddenHost.elementSize(),
            0.0f);


    // --------------------------------------------------------
    // FILL TIMESTEP
    // --------------------------------------------------------

    float* timestepData =
            timestepHost.host<float>();

    if (timestepData == nullptr) {

        report << "FAIL: Timestep host tensor allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }

    timestepData[0] = 0.0f;


    // --------------------------------------------------------
    // COPY INPUTS
    // --------------------------------------------------------

    if (!encoderInput->copyFromHostTensor(&encoderHost)) {

        report << "FAIL: Encoder input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (!hiddenInput->copyFromHostTensor(&hiddenHost)) {

        report << "FAIL: Hidden input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (!timestepInput->copyFromHostTensor(&timestepHost)) {

        report << "FAIL: Timestep input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // RUN
    // --------------------------------------------------------

    report << "Running inference...\n\n";

    auto start =
            std::chrono::steady_clock::now();

    MNN::ErrorCode result =
            interpreter->runSession(session);

    auto end =
            std::chrono::steady_clock::now();

    double elapsed =
            std::chrono::duration<double, std::milli>(
                    end - start).count();


    report << "Time: "
           << elapsed
           << " ms\n\n";

    report << "Error code: "
           << static_cast<int>(result)
           << "\n\n";


    if (result != MNN::NO_ERROR) {

        report << "FAIL: Transformer inference error.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // OUTPUT
    // --------------------------------------------------------

    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    "sample");

    if (output == nullptr) {

        report << "FAIL: Transformer output not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report << "Outputs: 1\n\n";

    report << "Output: sample\n";

    report << "Shape: "
           << formatShape(output)
           << "\n";

    report << "Elements: "
           << output->elementSize()
           << "\n\n";


    report << "PASS: Transformer executed successfully\n\n";


    destroyInterpreter(
            interpreter,
            session);


    report << "Transformer released successfully.\n";

    return report.str();
}


// ------------------------------------------------------------
// VAE SINGLE BACKEND TEST
// ------------------------------------------------------------

std::string runVaeOneBackend(
        const std::string& vaePath,
        bool useOpenCl) {

    std::ostringstream report;


    // --------------------------------------------------------
    // HEADER
    // --------------------------------------------------------

    if (useOpenCl) {

        report << "\n========================================\n";
        report << "VAE OPENCL TEST\n";
        report << "========================================\n\n";

    } else {

        report << "\n========================================\n";
        report << "VAE CPU TEST\n";
        report << "========================================\n\n";
    }


    report << "Model file:\n";
    report << vaePath << "\n\n";


    long long fileSize =
            getFileSize(vaePath);

    report << "File size: "
           << fileSize
           << " bytes\n\n";


    if (fileSize <= 0) {

        report << "FAIL: VAE file not found or empty.\n";

        return report.str();
    }


    // --------------------------------------------------------
    // CREATE INTERPRETER
    // --------------------------------------------------------

    report << "Creating ";

    if (useOpenCl) {
        report << "OpenCL";
    } else {
        report << "CPU";
    }

    report << " interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    if (interpreter == nullptr) {

        report << "FAIL: Could not create VAE interpreter.\n";

        return report.str();
    }


    report << "Interpreter created.\n\n";


    // --------------------------------------------------------
    // CREATE SESSION
    // --------------------------------------------------------

    report << "Creating session...\n";


    MNN::Session* session =
            createSession(
                    interpreter,
                    useOpenCl);


    if (session == nullptr) {

        report << "FAIL: Could not create VAE session.\n";

        interpreter->release();

        return report.str();
    }


    report << "Session created.\n\n";


    // --------------------------------------------------------
    // FIND INPUT
    // --------------------------------------------------------

    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        report << "Named input 'latent' not found.\n";
        report << "Trying first session input...\n";

        input =
                interpreter->getSessionInput(
                        session,
                        nullptr);
    }


    if (input == nullptr) {

        report << "FAIL: No VAE input tensor found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // INPUT INFO
    // --------------------------------------------------------

    report << "Device input shape: "
           << formatShape(input)
           << "\n";

    report << "Device input elements: "
           << input->elementSize()
           << "\n";

    report << "Device input type: "
           << static_cast<int>(input->getType().code)
           << "\n";

    report << "Device input bytes: "
           << static_cast<int>(input->getType().bytes())
           << "\n";

    report << "Device input format: "
           << dimensionTypeName(
                    input->getDimensionType())
           << "\n";

    report << "Device ID: "
           << input->deviceId()
           << "\n\n";


    // --------------------------------------------------------
    // EXPECTED SANA 512 LATENT
    // --------------------------------------------------------

    const std::vector<int> expectedShape = {
        1,
        32,
        16,
        16
    };


    bool shapeCorrect = true;


    if (input->dimensions() != 4) {

        shapeCorrect = false;

    } else {

        for (int i = 0; i < 4; ++i) {

            if (input->length(i) != expectedShape[i]) {

                shapeCorrect = false;

                break;
            }
        }
    }


    if (!shapeCorrect) {

        report << "Input shape is not [1,32,16,16].\n";

        report << "Resizing to [1,32,16,16]...\n";


        input->resize(
                expectedShape);


        interpreter->resizeTensor(
                input,
                expectedShape);


        interpreter->resizeSession(
                session);


        report << "Resize completed.\n\n";
    }


    // --------------------------------------------------------
    // FINAL SHAPE
    // --------------------------------------------------------

    report << "Final device shape: "
           << formatShape(input)
           << "\n\n";


    // --------------------------------------------------------
    // CREATE HOST TENSOR
    // --------------------------------------------------------

    report << "Creating simple MNN host tensor...\n";


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    report << "Host tensor shape: "
           << formatShape(&hostTensor)
           << "\n";

    report << "Host tensor elements: "
           << hostTensor.elementSize()
           << "\n";

    report << "Host tensor type: "
           << static_cast<int>(
                    hostTensor.getType().code)
           << "\n";

    report << "Host tensor bytes: "
           << static_cast<int>(
                    hostTensor.getType().bytes())
           << "\n";

    report << "Host tensor format: "
           << dimensionTypeName(
                    hostTensor.getDimensionType())
           << "\n";

    report << "Host tensor device ID: "
           << hostTensor.deviceId()
           << "\n\n";


    // --------------------------------------------------------
    // HOST POINTER
    // --------------------------------------------------------

    float* latent =
            hostTensor.host<float>();


    if (latent == nullptr) {

        report << "FAIL: Host memory allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report << "PASS: Host memory allocated.\n\n";


    // --------------------------------------------------------
    // FILL LATENT
    // --------------------------------------------------------

    /*
     * Deterministic test pattern.
     *
     * This is NOT a real Sana latent.
     * It is only used to test tensor transfer
     * and VAE execution.
     */

    const size_t elements =
            hostTensor.elementSize();


    for (size_t i = 0; i < elements; ++i) {

        latent[i] =
                -0.05f +
                static_cast<float>(i) * 0.000390625f;
    }


    // Keep values reasonable.

    for (size_t i = 0; i < elements; ++i) {

        if (latent[i] > 0.05f) {

            latent[i] = 0.05f;
        }
    }


    report << "Latent sample:\n";

    report << latent[0]
           << ", "
           << latent[1]
           << ", "
           << latent[2]
           << ", "
           << latent[3]
           << "\n\n";


    // --------------------------------------------------------
    // COPY TO DEVICE
    // --------------------------------------------------------

    report << "Calling device input copyFromHostTensor...\n\n";


    bool copyResult =
            input->copyFromHostTensor(
                    &hostTensor);


    report << "copyFromHostTensor: "
           << (copyResult ? "SUCCESS" : "FAILED")
           << "\n\n";


    if (!copyResult) {

        if (useOpenCl) {

            report << "RESULT: OpenCL VAE input copy failed.\n";

        } else {

            report << "RESULT: CPU VAE input copy failed.\n";
        }


        destroyInterpreter(
                interpreter,
                session);


        return report.str();
    }


    // --------------------------------------------------------
    // COPY SUCCESS
    // --------------------------------------------------------

    report << "PASS: VAE input copied successfully.\n\n";


    // --------------------------------------------------------
    // RUN VAE
    // --------------------------------------------------------

    report << "Running VAE inference...\n\n";


    auto start =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(session);


    auto end =
            std::chrono::steady_clock::now();


    double elapsed =
            std::chrono::duration<double, std::milli>(
                    end - start).count();


    report << "Time: "
           << elapsed
           << " ms\n\n";


    report << "Error code: "
           << static_cast<int>(result)
           << "\n\n";


    if (result != MNN::NO_ERROR) {

        report << "FAIL: VAE inference error.\n";


        destroyInterpreter(
                interpreter,
                session);


        return report.str();
    }


    // --------------------------------------------------------
    // GET OUTPUT
    // --------------------------------------------------------

    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report << "FAIL: VAE output tensor not found.\n";


        destroyInterpreter(
                interpreter,
                session);


        return report.str();
    }


    // --------------------------------------------------------
    // OUTPUT INFO
    // --------------------------------------------------------

    report << "VAE output shape: "
           << formatShape(output)
           << "\n";

    report << "VAE output elements: "
           << output->elementSize()
           << "\n";


    // --------------------------------------------------------
    // OUTPUT HOST TENSOR
    // --------------------------------------------------------

    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);


    bool outputCopy =
            outputHost.copyToHostTensor(output);


    if (!outputCopy) {

        report << "\nFAIL: Could not copy VAE output to host.\n";


        destroyInterpreter(
                interpreter,
                session);


        return report.str();
    }


    float* outputData =
            outputHost.host<float>();


    if (outputData != nullptr) {

        report << "\nOutput sample:\n";

        size_t outputElements =
                outputHost.elementSize();

        size_t sampleCount =
                std::min(
                        static_cast<size_t>(4),
                        outputElements);


        for (size_t i = 0;
             i < sampleCount;
             ++i) {

            if (i > 0) {
                report << ", ";
            }

            report << outputData[i];
        }

        report << "\n";
    }


    // --------------------------------------------------------
    // SUCCESS
    // --------------------------------------------------------

    if (useOpenCl) {

        report << "\nPASS: VAE executed successfully on OpenCL.\n";

    } else {

        report << "\nPASS: VAE executed successfully on CPU.\n";
    }


    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------

    destroyInterpreter(
            interpreter,
            session);


    report << "\nVAE interpreter released successfully.\n";


    return report.str();
}


// ------------------------------------------------------------
// VAE DUAL DIAGNOSTIC
// ------------------------------------------------------------

std::string runVaeDualDiagnostic(
        const std::string& vaePath) {

    std::ostringstream report;


    report << "SANA 0.6B / 512 VAE DUAL BACKEND TEST\n\n";

    report << "This diagnostic runs the same VAE model twice:\n";
    report << "1. CPU\n";
    report << "2. OpenCL\n\n";

    report << "The goal is to isolate whether the failure is:\n";
    report << "- VAE model/conversion related\n";
    report << "- or OpenCL/backend tensor transfer related\n\n";


    // --------------------------------------------------------
    // CPU FIRST
    // --------------------------------------------------------

    report << "Starting CPU VAE test...\n";

    report << runVaeOneBackend(
            vaePath,
            false);


    // --------------------------------------------------------
    // OPENCL SECOND
    // --------------------------------------------------------

    report << "\n\nStarting OpenCL VAE test...\n";

    report << runVaeOneBackend(
            vaePath,
            true);


    // --------------------------------------------------------
    // FINAL INTERPRETATION
    // --------------------------------------------------------

    report << "\n\n========================================\n";
    report << "VAE DIAGNOSTIC SUMMARY\n";
    report << "========================================\n\n";


    report << "Interpretation:\n\n";

    report << "CPU PASS + OpenCL FAIL:\n";
    report << "The VAE model can execute and the problem is isolated to the OpenCL/backend input path.\n\n";

    report << "CPU FAIL + OpenCL FAIL:\n";
    report << "The VAE model/conversion or MNN graph itself needs investigation.\n\n";

    report << "CPU PASS + OpenCL PASS:\n";
    report << "The VAE is working on both backends and we can move to the real Sana sampling pipeline.\n\n";

    report << "CPU FAIL + OpenCL PASS:\n";
    report << "The model works on OpenCL but has a CPU backend compatibility issue.\n";


    return report.str();
}


// ------------------------------------------------------------
// COPY ASSET TO FILE
// ------------------------------------------------------------

bool copyAssetToFile(
        AAssetManager* assetManager,
        const std::string& assetName,
        const std::string& outputPath) {

    if (assetManager == nullptr) {
        return false;
    }


    AAsset* asset =
            AAssetManager_open(
                    assetManager,
                    assetName.c_str(),
                    AASSET_MODE_STREAMING);


    if (asset == nullptr) {

        LOGE(
                "Could not open asset: %s",
                assetName.c_str());

        return false;
    }


    std::ofstream output(
            outputPath,
            std::ios::binary);


    if (!output.good()) {

        AAsset_close(asset);

        return false;
    }


    constexpr size_t bufferSize =
            1024 * 1024;


    std::vector<char> buffer(
            bufferSize);


    int bytesRead = 0;


    while ((bytesRead =
                    AAsset_read(
                            asset,
                            buffer.data(),
                            buffer.size())) > 0) {

        output.write(
                buffer.data(),
                bytesRead);
    }


    output.close();

    AAsset_close(asset);


    return bytesRead >= 0;
}

} // namespace


// ============================================================
// JNI
// ============================================================


// ------------------------------------------------------------
// INITIALIZE
// ------------------------------------------------------------

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv* env,
        jobject,
        jobject assetManagerObject,
        jstring modelAsset,
        jstring cachePath,
        jboolean preferOpenCl,
        jint cpuThreads) {

    std::lock_guard<std::mutex> lock(gMutex);


    releaseGlobalLocked();


    AAssetManager* assetManager =
            AAssetManager_fromJava(
                    env,
                    assetManagerObject);


    if (assetManager == nullptr) {

        gStatus =
                "Failed to obtain AssetManager";

        return JNI_FALSE;
    }


    std::string asset =
            jstringToString(
                    env,
                    modelAsset);


    std::string cache =
            jstringToString(
                    env,
                    cachePath);


    std::string modelPath =
            cache +
            "/sana_model.mnn";


    if (!copyAssetToFile(
                assetManager,
                asset,
                modelPath)) {

        gStatus =
                "Failed to copy model asset";

        return JNI_FALSE;
    }


    bool useOpenCl =
            preferOpenCl == JNI_TRUE;


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    modelPath.c_str());


    if (interpreter == nullptr) {

        gStatus =
                "Failed to create MNN interpreter";

        return JNI_FALSE;
    }


    MNN::ScheduleConfig config =
            makeScheduleConfig(
                    useOpenCl,
                    static_cast<int>(cpuThreads));


    MNN::Session* session =
            interpreter->createSession(
                    config);


    if (session == nullptr) {

        interpreter->release();

        gStatus =
                "Failed to create MNN session";

        return JNI_FALSE;
    }


    gInterpreter =
            interpreter;

    gSession =
            session;

    gInitialized =
            true;


    if (useOpenCl) {

        gBackend =
                "OpenCL / FP16";

    } else {

        gBackend =
                "CPU";
    }


    gStatus =
            "Initialized successfully";


    return JNI_TRUE;
}


// ------------------------------------------------------------
// IS INITIALIZED
// ------------------------------------------------------------

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(gMutex);

    return gInitialized
            ? JNI_TRUE
            : JNI_FALSE;
}


// ------------------------------------------------------------
// GET BACKEND
// ------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject) {

    std::lock_guard<std::mutex> lock(gMutex);

    return env->NewStringUTF(
            gBackend.c_str());
}


// ------------------------------------------------------------
// GET STATUS
// ------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject) {

    std::lock_guard<std::mutex> lock(gMutex);

    return env->NewStringUTF(
            gStatus.c_str());
}


// ------------------------------------------------------------
// RELEASE
// ------------------------------------------------------------

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(gMutex);

    releaseGlobalLocked();
}


// ------------------------------------------------------------
// TEST TRANSFORMER
// ------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
        JNIEnv* env,
        jobject,
        jstring transformerPath,
        jstring cachePath,
        jboolean preferOpenCl) {

    std::string transformer =
            jstringToString(
                    env,
                    transformerPath);


    std::string cache =
            jstringToString(
                    env,
                    cachePath);


    bool useOpenCl =
            preferOpenCl == JNI_TRUE;


    std::string result =
            runTransformerDiagnostic(
                    transformer,
                    cache,
                    useOpenCl);


    return env->NewStringUTF(
            result.c_str());
}


// ------------------------------------------------------------
// TEST VAE
// ------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl) {

    /*
     * IMPORTANT:
     *
     * The Kotlin API stays unchanged.
     *
     * We intentionally ignore preferOpenCl here because
     * this diagnostic needs BOTH backends to isolate the
     * VAE problem.
     */

    (void)cachePath;
    (void)preferOpenCl;


    std::string vae =
            jstringToString(
                    env,
                    vaePath);


    std::string result =
            runVaeDualDiagnostic(
                    vae);


    return env->NewStringUTF(
            result.c_str());
}


// ------------------------------------------------------------
// TEST BOTH MODELS
// ------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestModels(
        JNIEnv* env,
        jobject,
        jstring transformerPath,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl) {

    std::string transformer =
            jstringToString(
                    env,
                    transformerPath);


    std::string vae =
            jstringToString(
                    env,
                    vaePath);


    std::string cache =
            jstringToString(
                    env,
                    cachePath);


    bool useOpenCl =
            preferOpenCl == JNI_TRUE;


    std::ostringstream report;


    // --------------------------------------------------------
    // TRANSFORMER
    // --------------------------------------------------------

    report << runTransformerDiagnostic(
            transformer,
            cache,
            useOpenCl);


    // --------------------------------------------------------
    // VAE
    // --------------------------------------------------------

    report << "\n\n";


    report << runVaeDualDiagnostic(
            vae);


    return env->NewStringUTF(
            report.str().c_str());
}
