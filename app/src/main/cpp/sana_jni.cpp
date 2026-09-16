#include <jni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
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


// ============================================================
// HELPERS
// ============================================================

std::string jstringToString(
        JNIEnv* env,
        jstring value) {

    if (value == nullptr) {
        return "";
    }

    const char* chars =
            env->GetStringUTFChars(
                    value,
                    nullptr);

    if (chars == nullptr) {
        return "";
    }

    std::string result(chars);

    env->ReleaseStringUTFChars(
            value,
            chars);

    return result;
}


std::string formatShape(
        const MNN::Tensor* tensor) {

    if (tensor == nullptr) {
        return "null";
    }

    std::ostringstream out;

    out << "[";

    for (int i = 0;
         i < tensor->dimensions();
         ++i) {

        if (i > 0) {
            out << ", ";
        }

        out << tensor->length(i);
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
            return "CAFFE_C4";

        default:
            return "UNKNOWN";
    }
}


long long getFileSize(
        const std::string& path) {

    std::ifstream file(
            path,
            std::ios::binary |
            std::ios::ate);

    if (!file.good()) {
        return -1;
    }

    return static_cast<long long>(
            file.tellg());
}


// ============================================================
// MNN SESSION
// ============================================================

MNN::ScheduleConfig makeScheduleConfig(
        bool preferOpenCl,
        int cpuThreads) {

    MNN::ScheduleConfig config;

    if (preferOpenCl) {

        config.type =
                MNN_FORWARD_OPENCL;

        config.mode =
                MNN_GPU_TUNING_HEAVY;

    } else {

        config.type =
                MNN_FORWARD_CPU;

        config.numThread =
                cpuThreads;
    }

    return config;
}


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

    if (preferOpenCl) {

        config.backupType =
                MNN_FORWARD_CPU;
    }

    return interpreter->createSession(
            config);
}


// ============================================================
// CLEANUP
// ============================================================

void destroyInterpreter(
        MNN::Interpreter*& interpreter,
        MNN::Session*& session) {

    if (interpreter == nullptr) {
        return;
    }

    if (session != nullptr) {

        interpreter->releaseSession(
                session);

        session = nullptr;
    }

    /*
     * This MNN version does not expose
     * Interpreter::release().
     *
     * The interpreter returned by
     * createFromFile() is destroyed
     * with delete.
     */

    delete interpreter;

    interpreter = nullptr;
}


void releaseGlobalLocked() {

    if (gInterpreter != nullptr) {

        if (gSession != nullptr) {

            gInterpreter->releaseSession(
                    gSession);

            gSession = nullptr;
        }

        delete gInterpreter;

        gInterpreter = nullptr;
    }

    gInitialized = false;

    gBackend =
            "Not initialized";

    gStatus =
            "Released";
}


// ============================================================
// TRANSFORMER DIAGNOSTIC
// ============================================================

std::string runTransformerDiagnostic(
        const std::string& transformerPath,
        const std::string& cachePath,
        bool preferOpenCl) {

    (void)cachePath;

    std::ostringstream report;

    report
            << "SANA 0.6B / 512 TRANSFORMER TEST\n\n";

    report
            << "Transformer file:\n"
            << transformerPath
            << "\n\n";


    long long fileSize =
            getFileSize(
                    transformerPath);

    report
            << "File size: "
            << fileSize
            << " bytes\n\n";


    if (fileSize <= 0) {

        report
                << "FAIL: Transformer file not found or empty.\n";

        return report.str();
    }


    // --------------------------------------------------------
    // INTERPRETER
    // --------------------------------------------------------

    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    transformerPath.c_str());


    if (interpreter == nullptr) {

        report
                << "FAIL: Could not create Transformer interpreter.\n";

        return report.str();
    }


    report
            << "Transformer interpreter created.\n\n";


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

    MNN::Session* session =
            createSession(
                    interpreter,
                    preferOpenCl);


    if (session == nullptr) {

        report
                << "FAIL: Could not create Transformer session.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (preferOpenCl) {

        report
                << "Backend: OpenCL / FP16\n";

    } else {

        report
                << "Backend: CPU\n";
    }


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

        report
                << "FAIL: Required Transformer inputs not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // LOG INPUTS
    // --------------------------------------------------------

    report
            << "\nInputs: 3\n\n";


    report
            << "Input: encoder_hidden_states\n"
            << "Shape: "
            << formatShape(
                    encoderInput)
            << "\n"
            << "Elements: "
            << encoderInput->elementSize()
            << "\n\n";


    report
            << "Input: hidden_states\n"
            << "Shape: "
            << formatShape(
                    hiddenInput)
            << "\n"
            << "Elements: "
            << hiddenInput->elementSize()
            << "\n\n";


    report
            << "Input: timestep\n"
            << "Shape: "
            << formatShape(
                    timestepInput)
            << "\n"
            << "Elements: "
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
    // HOST POINTERS
    // --------------------------------------------------------

    float* encoderData =
            encoderHost.host<float>();


    float* hiddenData =
            hiddenHost.host<float>();


    float* timestepData =
            timestepHost.host<float>();


    if (encoderData == nullptr ||
        hiddenData == nullptr ||
        timestepData == nullptr) {

        report
                << "FAIL: Transformer host allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // TEST VALUES
    // --------------------------------------------------------

    std::fill(
            encoderData,
            encoderData +
            encoderHost.elementSize(),
            0.0f);


    std::fill(
            hiddenData,
            hiddenData +
            hiddenHost.elementSize(),
            0.0f);


    timestepData[0] =
            0.0f;


    // --------------------------------------------------------
    // COPY
    // --------------------------------------------------------

    if (!encoderInput->copyFromHostTensor(
                &encoderHost)) {

        report
                << "FAIL: Encoder input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (!hiddenInput->copyFromHostTensor(
                &hiddenHost)) {

        report
                << "FAIL: Hidden input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (!timestepInput->copyFromHostTensor(
                &timestepHost)) {

        report
                << "FAIL: Timestep input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // RUN
    // --------------------------------------------------------

    report
            << "Running inference...\n\n";


    auto start =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto end =
            std::chrono::steady_clock::now();


    double elapsed =
            std::chrono::duration<double, std::milli>(
                    end - start)
            .count();


    report
            << "Time: "
            << elapsed
            << " ms\n\n";


    report
            << "Error code: "
            << static_cast<int>(
                    result)
            << "\n\n";


    if (result != MNN::NO_ERROR) {

        report
                << "FAIL: Transformer inference error.\n";

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

        report
                << "FAIL: Transformer output not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Outputs: 1\n\n"
            << "Output: sample\n"
            << "Shape: "
            << formatShape(output)
            << "\n"
            << "Elements: "
            << output->elementSize()
            << "\n\n";


    report
            << "PASS: Transformer executed successfully\n\n";


    destroyInterpreter(
            interpreter,
            session);


    report
            << "Transformer released successfully.\n";


    return report.str();
}


// ============================================================
// VAE TEST - ONE BACKEND
// ============================================================

std::string runVaeOneBackend(
        const std::string& vaePath,
        bool useOpenCl) {

    std::ostringstream report;


    if (useOpenCl) {

        report
                << "\n========================================\n"
                << "VAE OPENCL TEST\n"
                << "========================================\n\n";

    } else {

        report
                << "\n========================================\n"
                << "VAE CPU TEST\n"
                << "========================================\n\n";
    }


    report
            << "Model file:\n"
            << vaePath
            << "\n\n";


    long long fileSize =
            getFileSize(
                    vaePath);


    report
            << "File size: "
            << fileSize
            << " bytes\n\n";


    if (fileSize <= 0) {

        report
                << "FAIL: VAE file not found or empty.\n";

        return report.str();
    }


    // --------------------------------------------------------
    // INTERPRETER
    // --------------------------------------------------------

    report
            << "Creating ";

    if (useOpenCl) {
        report << "OpenCL";
    } else {
        report << "CPU";
    }

    report
            << " interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    if (interpreter == nullptr) {

        report
                << "FAIL: Could not create VAE interpreter.\n";

        return report.str();
    }


    report
            << "Interpreter created.\n\n";


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

    report
            << "Creating session...\n";


    MNN::Session* session =
            createSession(
                    interpreter,
                    useOpenCl);


    if (session == nullptr) {

        report
                << "FAIL: Could not create VAE session.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Session created.\n\n";


    // --------------------------------------------------------
    // INPUT
    // --------------------------------------------------------

    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        report
                << "Named input 'latent' not found.\n";

        report
                << "Trying first session input...\n";


        input =
                interpreter->getSessionInput(
                        session,
                        nullptr);
    }


    if (input == nullptr) {

        report
                << "FAIL: No VAE input tensor found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // INPUT INFO
    // --------------------------------------------------------

    report
            << "Device input shape: "
            << formatShape(input)
            << "\n";


    report
            << "Device input elements: "
            << input->elementSize()
            << "\n";


    report
            << "Device input type: "
            << static_cast<int>(
                    input->getType().code)
            << "\n";


    report
            << "Device input bytes: "
            << static_cast<int>(
                    input->getType().bytes())
            << "\n";


    report
            << "Device input format: "
            << dimensionTypeName(
                    input->getDimensionType())
            << "\n";


    report
            << "Device ID: "
            << input->deviceId()
            << "\n\n";


    // --------------------------------------------------------
    // IMPORTANT:
    // DO NOT RESIZE.
    //
    // Sana 512 uses:
    // [1, 32, 16, 16]
    // --------------------------------------------------------

    bool shapeCorrect =
            input->dimensions() == 4 &&
            input->length(0) == 1 &&
            input->length(1) == 32 &&
            input->length(2) == 16 &&
            input->length(3) == 16;


    if (!shapeCorrect) {

        report
                << "FAIL: Unexpected VAE input shape.\n";

        report
                << "Expected: [1,32,16,16]\n";

        report
                << "Actual: "
                << formatShape(input)
                << "\n";


        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "PASS: VAE input shape is [1,32,16,16].\n\n";


    // --------------------------------------------------------
    // HOST TENSOR
    // --------------------------------------------------------

    report
            << "Creating simple MNN host tensor...\n";


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    report
            << "Host tensor shape: "
            << formatShape(
                    &hostTensor)
            << "\n";


    report
            << "Host tensor elements: "
            << hostTensor.elementSize()
            << "\n";


    report
            << "Host tensor type: "
            << static_cast<int>(
                    hostTensor.getType().code)
            << "\n";


    report
            << "Host tensor bytes: "
            << static_cast<int>(
                    hostTensor.getType().bytes())
            << "\n";


    report
            << "Host tensor format: "
            << dimensionTypeName(
                    hostTensor.getDimensionType())
            << "\n";


    report
            << "Host tensor device ID: "
            << hostTensor.deviceId()
            << "\n\n";


    // --------------------------------------------------------
    // HOST POINTER
    // --------------------------------------------------------

    float* latent =
            hostTensor.host<float>();


    if (latent == nullptr) {

        report
                << "FAIL: Host memory allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "PASS: Host memory allocated.\n\n";


    // --------------------------------------------------------
    // FILL LATENT
    // --------------------------------------------------------

    size_t elements =
            hostTensor.elementSize();


    for (size_t i = 0;
         i < elements;
         ++i) {

        latent[i] =
                -0.05f +
                static_cast<float>(i) *
                0.000390625f;


        if (latent[i] > 0.05f) {
            latent[i] = 0.05f;
        }
    }


    report
            << "Latent sample:\n";


    size_t sampleCount =
            std::min(
                    static_cast<size_t>(4),
                    elements);


    for (size_t i = 0;
         i < sampleCount;
         ++i) {

        if (i > 0) {
            report << ", ";
        }

        report << latent[i];
    }


    report
            << "\n\n";


    // --------------------------------------------------------
    // COPY
    // --------------------------------------------------------

    report
            << "Calling device input copyFromHostTensor...\n\n";


    bool copyResult =
            input->copyFromHostTensor(
                    &hostTensor);


    report
            << "copyFromHostTensor: "
            << (copyResult
                    ? "SUCCESS"
                    : "FAILED")
            << "\n\n";


    if (!copyResult) {

        if (useOpenCl) {

            report
                    << "RESULT: OpenCL VAE input copy failed.\n";

        } else {

            report
                    << "RESULT: CPU VAE input copy failed.\n";
        }


        destroyInterpreter(
                interpreter,
                session);


        return report.str();
    }


    report
            << "PASS: VAE input copied successfully.\n\n";


    // --------------------------------------------------------
    // RUN
    // --------------------------------------------------------

    report
            << "Running VAE inference...\n\n";


    auto start =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto end =
            std::chrono::steady_clock::now();


    double elapsed =
            std::chrono::duration<double, std::milli>(
                    end - start)
            .count();


    report
            << "Time: "
            << elapsed
            << " ms\n\n";


    report
            << "Error code: "
            << static_cast<int>(
                    result)
            << "\n\n";


    if (result != MNN::NO_ERROR) {

        report
                << "FAIL: VAE inference error.\n";

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
                    nullptr);


    if (output == nullptr) {

        report
                << "FAIL: VAE output tensor not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "VAE output shape: "
            << formatShape(output)
            << "\n";


    report
            << "VAE output elements: "
            << output->elementSize()
            << "\n\n";


    // --------------------------------------------------------
    // OUTPUT COPY
    // --------------------------------------------------------

    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);


    bool outputCopy =
            output->copyToHostTensor(
                    &outputHost);


    if (!outputCopy) {

        report
                << "FAIL: Could not copy VAE output to host.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    float* outputData =
            outputHost.host<float>();


    if (outputData != nullptr) {

        report
                << "Output sample:\n";


        size_t outputElements =
                outputHost.elementSize();


        size_t count =
                std::min(
                        static_cast<size_t>(4),
                        outputElements);


        for (size_t i = 0;
             i < count;
             ++i) {

            if (i > 0) {
                report << ", ";
            }

            report
                    << outputData[i];
        }


        report
                << "\n\n";
    }


    // --------------------------------------------------------
    // SUCCESS
    // --------------------------------------------------------

    if (useOpenCl) {

        report
                << "PASS: VAE executed successfully on OpenCL.\n";

    } else {

        report
                << "PASS: VAE executed successfully on CPU.\n";
    }


    destroyInterpreter(
            interpreter,
            session);


    report
            << "\nVAE interpreter released successfully.\n";


    return report.str();
}


// ============================================================
// VAE CPU + OPENCL
// ============================================================

std::string runVaeDualDiagnostic(
        const std::string& vaePath) {

    std::ostringstream report;


    report
            << "SANA 0.6B / 512 VAE DUAL BACKEND TEST\n\n";


    report
            << "Testing the same VAE model on:\n"
            << "1. CPU\n"
            << "2. OpenCL\n\n";


    report
            << "Purpose:\n"
            << "Determine whether the VAE problem is model-related "
            << "or OpenCL/backend-specific.\n";


    // --------------------------------------------------------
    // CPU
    // --------------------------------------------------------

    report
            << "\n\nStarting CPU VAE test...\n";


    std::string cpuResult =
            runVaeOneBackend(
                    vaePath,
                    false);


    report
            << cpuResult;


    // --------------------------------------------------------
    // OPENCL
    // --------------------------------------------------------

    report
            << "\n\nStarting OpenCL VAE test...\n";


    std::string openClResult =
            runVaeOneBackend(
                    vaePath,
                    true);


    report
            << openClResult;


    // --------------------------------------------------------
    // SUMMARY
    // --------------------------------------------------------

    report
            << "\n\n========================================\n"
            << "VAE DIAGNOSTIC SUMMARY\n"
            << "========================================\n\n";


    bool cpuPass =
            cpuResult.find(
                    "PASS: VAE executed successfully on CPU.")
            != std::string::npos;


    bool openClPass =
            openClResult.find(
                    "PASS: VAE executed successfully on OpenCL.")
            != std::string::npos;


    if (cpuPass && openClPass) {

        report
                << "CPU: PASS\n"
                << "OpenCL: PASS\n\n";

        report
                << "RESULT: VAE works on both backends.\n";

    } else if (cpuPass && !openClPass) {

        report
                << "CPU: PASS\n"
                << "OpenCL: FAIL\n\n";

        report
                << "RESULT: VAE model executes on CPU, "
                << "but OpenCL path is failing.\n";

    } else if (!cpuPass && !openClPass) {

        report
                << "CPU: FAIL\n"
                << "OpenCL: FAIL\n\n";

        report
                << "RESULT: VAE model/conversion or common "
                << "MNN graph path needs investigation.\n";

    } else {

        report
                << "CPU: FAIL\n"
                << "OpenCL: PASS\n\n";

        report
                << "RESULT: VAE works on OpenCL but not CPU.\n";
    }


    return report.str();
}


// ============================================================
// ASSET COPY
// ============================================================

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
// JNI INITIALIZE
// ============================================================

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

    std::lock_guard<std::mutex> lock(
            gMutex);


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
                    static_cast<int>(
                            cpuThreads));


    MNN::Session* session =
            interpreter->createSession(
                    config);


    if (session == nullptr) {

        destroyInterpreter(
                interpreter,
                session);

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


// ============================================================
// JNI IS INITIALIZED
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    return gInitialized
            ? JNI_TRUE
            : JNI_FALSE;
}


// ============================================================
// JNI BACKEND
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    return env->NewStringUTF(
            gBackend.c_str());
}


// ============================================================
// JNI STATUS
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    return env->NewStringUTF(
            gStatus.c_str());
}


// ============================================================
// JNI RELEASE
// ============================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    releaseGlobalLocked();
}


// ============================================================
// JNI TRANSFORMER TEST
// ============================================================

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


// ============================================================
// JNI VAE TEST
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl) {

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


// ============================================================
// JNI BOTH MODELS
// ============================================================

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

    report
            << runTransformerDiagnostic(
                    transformer,
                    cache,
                    useOpenCl);


    // --------------------------------------------------------
    // VAE
    // --------------------------------------------------------

    report
            << "\n\n";


    report
            << runVaeDualDiagnostic(
                    vae);


    return env->NewStringUTF(
            report.str().c_str());
}
