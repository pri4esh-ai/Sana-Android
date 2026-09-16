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


    // IMPORTANT:
    // Host tensors are derived from their device tensors.

    MNN::Tensor encoderHost(
            encoderInput,
            MNN::Tensor::CAFFE);

    MNN::Tensor hiddenHost(
            hiddenInput,
            MNN::Tensor::CAFFE);

    MNN::Tensor timestepHost(
            timestepInput,
            MNN::Tensor::CAFFE);


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

    timestepData[0] = 0.0f;


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
// VAE - ONE BACKEND
// ============================================================

std::string runVaeOneBackend(
        const std::string& vaePath,
        bool useOpenCl) {

    std::ostringstream report;

    report
            << "\n========================================\n";

    if (useOpenCl) {

        report
                << "VAE OPENCL TEST\n";

    } else {

        report
                << "VAE CPU TEST\n";
    }

    report
            << "========================================\n\n";


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
            << "\n\n";

    report
            << "Device input elements: "
            << input->elementSize()
            << "\n\n";

    report
            << "Device input type: "
            << static_cast<int>(
                    input->getType().code)
            << "\n\n";

    report
            << "Device input bytes: "
            << input->getType().bytes()
            << "\n\n";

    report
            << "Device input format: "
            << dimensionTypeName(
                    input->getDimensionType())
            << "\n\n";

    report
            << "Device ID: "
            << input->deviceId()
            << "\n\n";


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
    // IMPORTANT HOST TENSOR TEST
    // --------------------------------------------------------

    report
            << "Creating device-derived MNN host tensor...\n\n";


    /*
     * IMPORTANT:
     *
     * Do NOT create an unrelated tensor with:
     *
     *     MNN::Tensor(4, CAFFE)
     *
     * Instead derive the host tensor directly from the
     * session input tensor.
     */

    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    report
            << "Host tensor shape: "
            << formatShape(
                    &hostTensor)
            << "\n\n";

    report
            << "Host tensor elements: "
            << hostTensor.elementSize()
            << "\n\n";

    report
            << "Host tensor type: "
            << static_cast<int>(
                    hostTensor.getType().code)
            << "\n\n";

    report
            << "Host tensor bytes: "
            << hostTensor.getType().bytes()
            << "\n\n";

    report
            << "Host tensor format: "
            << dimensionTypeName(
                    hostTensor.getDimensionType())
            << "\n\n";

    report
            << "Host tensor device ID: "
            << hostTensor.deviceId()
            << "\n\n";


    float* latentData =
            hostTensor.host<float>();


    if (latentData == nullptr) {

        report
                << "FAIL: Host tensor pointer is null.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "PASS: Host memory allocated.\n\n";


    // --------------------------------------------------------
    // LATENT DATA
    // --------------------------------------------------------

    for (int i = 0;
         i < hostTensor.elementSize();
         ++i) {

        /*
         * Small deterministic latent values.
         *
         * This is only a transport/inference diagnostic.
         * It is NOT intended to produce a meaningful image.
         */

        latentData[i] =
                -0.05f +
                (
                    static_cast<float>(
                        i % 256)
                    / 255.0f
                ) * 0.10f;
    }


    report
            << "Latent sample:\n";

    int sampleCount =
            std::min(
                    4,
                    hostTensor.elementSize());

    for (int i = 0;
         i < sampleCount;
         ++i) {

        if (i > 0) {
            report << ", ";
        }

        report
                << latentData[i];
    }

    report
            << "\n\n";


    // --------------------------------------------------------
    // DEVICE COPY
    // --------------------------------------------------------

    report
            << "Calling device input copyFromHostTensor...\n";


    bool copied =
            input->copyFromHostTensor(
                    &hostTensor);


    report
            << "copyFromHostTensor: "
            << (
                copied
                ? "SUCCESS"
                : "FAILED"
            )
            << "\n\n";


    if (!copied) {

        report
                << "RESULT: ";

        if (useOpenCl) {
            report << "OpenCL";
        } else {
            report << "CPU";
        }

        report
                << " VAE input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    // --------------------------------------------------------
    // RUN VAE
    // --------------------------------------------------------

    report
            << "PASS: VAE input copied successfully.\n\n";

    report
            << "Running VAE inference...\n";


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
            << "VAE inference time: "
            << elapsed
            << " ms\n\n";

    report
            << "VAE error code: "
            << static_cast<int>(
                    result)
            << "\n\n";


    if (result != MNN::NO_ERROR) {

        report
                << "RESULT: VAE inference FAILED.\n";

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
                << "RESULT: VAE output tensor not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "VAE output shape: "
            << formatShape(output)
            << "\n\n";

    report
            << "VAE output elements: "
            << output->elementSize()
            << "\n\n";


    report
            << "PASS: VAE executed successfully.\n";


    destroyInterpreter(
            interpreter,
            session);


    report
            << "VAE released successfully.\n";


    return report.str();
}


// ============================================================
// VAE DUAL DIAGNOSTIC
// ============================================================

std::string runVaeDualDiagnostic(
        const std::string& vaePath) {

    std::ostringstream report;


    report
            << "SANA 0.6B / 512 VAE DIAGNOSTIC\n\n";


    report
            << "Only sana_vae_decoder.mnn is tested.\n";

    report
            << "The Transformer is completely excluded.\n\n";


    // CPU

    report
            << "Starting CPU VAE test...\n";

    std::string cpuResult =
            runVaeOneBackend(
                    vaePath,
                    false);

    report
            << cpuResult
            << "\n";


    // OpenCL

    report
            << "Starting OpenCL VAE test...\n";

    std::string openClResult =
            runVaeOneBackend(
                    vaePath,
                    true);

    report
            << openClResult
            << "\n";


    // Summary

    bool cpuPass =
            cpuResult.find(
                    "PASS: VAE executed successfully.")
            != std::string::npos;

    bool openClPass =
            openClResult.find(
                    "PASS: VAE executed successfully.")
            != std::string::npos;


    report
            << "\n========================================\n"
            << "VAE DIAGNOSTIC SUMMARY\n"
            << "========================================\n\n";


    report
            << "CPU: "
            << (
                cpuPass
                ? "PASS"
                : "FAIL"
            )
            << "\n";


    report
            << "OpenCL: "
            << (
                openClPass
                ? "PASS"
                : "FAIL"
            )
            << "\n\n";


    if (cpuPass && openClPass) {

        report
                << "RESULT: VAE works on CPU and OpenCL.\n";

    } else if (cpuPass && !openClPass) {

        report
                << "RESULT: VAE CPU works; OpenCL path still needs investigation.\n";

    } else if (!cpuPass && openClPass) {

        report
                << "RESULT: VAE OpenCL works; CPU backend has an issue.\n";

    } else {

        report
                << "RESULT: Both VAE paths failed.\n";

        report
                << "The next investigation point is the MNN VAE graph/conversion/input transport.\n";
    }


    return report.str();
}


// ============================================================
// JNI - INITIALIZE
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
        jint cpuThreads) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    releaseGlobalLocked();


    if (assetManager == nullptr ||
        modelAsset == nullptr) {

        gStatus =
                "Invalid initialization arguments";

        return JNI_FALSE;
    }


    AAssetManager* manager =
            AAssetManager_fromJava(
                    env,
                    assetManager);


    if (manager == nullptr) {

        gStatus =
                "Could not access AssetManager";

        return JNI_FALSE;
    }


    std::string assetName =
            jstringToString(
                    env,
                    modelAsset);

    std::string cache =
            jstringToString(
                    env,
                    cachePath);


    if (assetName.empty() ||
        cache.empty()) {

        gStatus =
                "Invalid model asset or cache path";

        return JNI_FALSE;
    }


    // --------------------------------------------------------
    // MODEL ASSET
    // --------------------------------------------------------

    AAsset* asset =
            AAssetManager_open(
                    manager,
                    assetName.c_str(),
                    AASSET_MODE_STREAMING);


    if (asset == nullptr) {

        gStatus =
                "Model asset not found";

        return JNI_FALSE;
    }


    std::string modelPath =
            cache +
            "/sana_model.mnn";


    std::ofstream output(
            modelPath,
            std::ios::binary);


    if (!output.good()) {

        AAsset_close(asset);

        gStatus =
                "Could not create cached model";

        return JNI_FALSE;
    }


    char buffer[64 * 1024];

    int count;


    while (
            (count =
                AAsset_read(
                        asset,
                        buffer,
                        sizeof(buffer)))
            > 0) {

        output.write(
                buffer,
                count);
    }


    output.close();

    AAsset_close(asset);


    // --------------------------------------------------------
    // INTERPRETER
    // --------------------------------------------------------

    gInterpreter =
            MNN::Interpreter::createFromFile(
                    modelPath.c_str());


    if (gInterpreter == nullptr) {

        gStatus =
                "Could not create MNN interpreter";

        return JNI_FALSE;
    }


    MNN::ScheduleConfig config =
            makeScheduleConfig(
                    preferOpenCl,
                    static_cast<int>(
                            cpuThreads));


    if (preferOpenCl) {

        config.backupType =
                MNN_FORWARD_CPU;
    }


    gSession =
            gInterpreter->createSession(
                    config);


    if (gSession == nullptr) {

        delete gInterpreter;

        gInterpreter = nullptr;

        gStatus =
                "Could not create MNN session";

        return JNI_FALSE;
    }


    gInitialized = true;


    if (preferOpenCl) {

        gBackend =
                "OpenCL / FP16";

    } else {

        gBackend =
                "CPU";
    }


    gStatus =
            "Sana model initialized";


    return JNI_TRUE;
}


// ============================================================
// JNI - STATE
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
// JNI - RELEASE
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
// JNI - TRANSFORMER TEST
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

    std::string result =
            runTransformerDiagnostic(
                    transformer,
                    cache,
                    preferOpenCl);

    return env->NewStringUTF(
            result.c_str());
}


// ============================================================
// JNI - VAE TEST
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
// JNI - BOTH MODELS
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


    std::ostringstream report;


    report
            << "SANA MODEL DIAGNOSTIC\n"
            << "=====================\n\n";


    report
            << runTransformerDiagnostic(
                    transformer,
                    cache,
                    preferOpenCl)
            << "\n";


    report
            << runVaeDualDiagnostic(
                    vae);


    return env->NewStringUTF(
            report.str().c_str());
}

} // namespace
