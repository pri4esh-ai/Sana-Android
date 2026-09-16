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

std::string jstringToString(
        JNIEnv* env,
        jstring value) {

    if (value == nullptr) {
        return "";
    }

    const char* chars =
            env->GetStringUTFChars(value, nullptr);

    if (chars == nullptr) {
        return "";
    }

    std::string result(chars);

    env->ReleaseStringUTFChars(value, chars);

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
// MNN SESSION CONFIG
// ============================================================

MNN::ScheduleConfig makeScheduleConfig(
        bool useOpenCl,
        int cpuThreads) {

    MNN::ScheduleConfig config;

    if (useOpenCl) {

        config.type =
                MNN_FORWARD_OPENCL;

        config.mode =
                MNN_GPU_TUNING_HEAVY;

        config.backupType =
                MNN_FORWARD_CPU;

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
        bool useOpenCl,
        int cpuThreads = 4) {

    if (interpreter == nullptr) {
        return nullptr;
    }

    MNN::ScheduleConfig config =
            makeScheduleConfig(
                    useOpenCl,
                    cpuThreads);

    return interpreter->createSession(config);
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

        interpreter->releaseSession(session);

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
// GENERIC VAE CPU TEST
// ============================================================

std::string runVaeCpuTest(
        const std::string& vaePath) {

    std::ostringstream report;

    report
            << "========================================\n"
            << "SANA VAE FP16 CPU TEST\n"
            << "========================================\n\n";


    // --------------------------------------------------------
    // FILE
    // --------------------------------------------------------

    report
            << "Model:\n"
            << vaePath
            << "\n\n";


    long long fileSize =
            getFileSize(vaePath);


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
            << "Creating MNN interpreter...\n";


    auto startInterpreter =
            std::chrono::steady_clock::now();


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    auto endInterpreter =
            std::chrono::steady_clock::now();


    double interpreterMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                            endInterpreter -
                            startInterpreter)
            .count();


    report
            << "Interpreter creation time: "
            << interpreterMs
            << " ms\n";


    if (interpreter == nullptr) {

        report
                << "FAIL: Could not create VAE interpreter.\n";

        return report.str();
    }


    report
            << "Interpreter: PASS\n\n";


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

    report
            << "Creating CPU session...\n";


    auto startSession =
            std::chrono::steady_clock::now();


    MNN::Session* session =
            createSession(
                    interpreter,
                    false,
                    4);


    auto endSession =
            std::chrono::steady_clock::now();


    double sessionMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                            endSession -
                            startSession)
            .count();


    report
            << "Session creation time: "
            << sessionMs
            << " ms\n";


    if (session == nullptr) {

        report
                << "FAIL: Could not create CPU session.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "CPU session: PASS\n\n";


    // --------------------------------------------------------
    // INPUT
    // --------------------------------------------------------

    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        input =
                interpreter->getSessionInput(
                        session,
                        nullptr);
    }


    if (input == nullptr) {

        report
                << "FAIL: VAE input not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Input shape: "
            << formatShape(input)
            << "\n";


    report
            << "Input elements: "
            << input->elementSize()
            << "\n";


    report
            << "Input type bytes: "
            << input->getType().bytes()
            << "\n";


    report
            << "Input format: "
            << dimensionTypeName(
                    input->getDimensionType())
            << "\n\n";


    // --------------------------------------------------------
    // EXACT SANA LATENT SHAPE
    // --------------------------------------------------------

    bool correctShape =
            input->dimensions() == 4 &&
            input->length(0) == 1 &&
            input->length(1) == 32 &&
            input->length(2) == 16 &&
            input->length(3) == 16;


    if (!correctShape) {

        report
                << "FAIL: Wrong VAE input shape.\n";

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
            << "Input shape: PASS\n\n";


    // --------------------------------------------------------
    // HOST TENSOR
    // --------------------------------------------------------

    report
            << "Creating host tensor...\n";


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE,
            true);


    float* data =
            hostTensor.host<float>();


    if (data == nullptr) {

        report
                << "FAIL: Host tensor allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Host tensor: PASS\n";


    // --------------------------------------------------------
    // LATENT DATA
    // --------------------------------------------------------

    const int elements =
            hostTensor.elementSize();


    for (int i = 0;
         i < elements;
         ++i) {

        data[i] =
                -0.05f +
                (
                    static_cast<float>(
                            i % 256)
                    / 255.0f
                ) * 0.10f;
    }


    report
            << "Latent sample: "
            << data[0]
            << ", "
            << data[1]
            << ", "
            << data[2]
            << ", "
            << data[3]
            << "\n\n";


    // --------------------------------------------------------
    // HOST -> DEVICE
    // --------------------------------------------------------

    report
            << "Copying latent to MNN...\n";


    bool copied =
            input->copyFromHostTensor(
                    &hostTensor);


    if (!copied) {

        report
                << "FAIL: Host -> device copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Host -> MNN: PASS\n\n";


    // --------------------------------------------------------
    // INFERENCE
    // --------------------------------------------------------

    report
            << "Running VAE inference...\n";


    auto startInference =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto endInference =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                            endInference -
                            startInference)
            .count();


    report
            << "Inference time: "
            << inferenceMs
            << " ms\n";


    report
            << "MNN error code: "
            << static_cast<int>(
                    result)
            << "\n\n";


    if (result != MNN::NO_ERROR) {

        report
                << "FAIL: VAE inference failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Inference: PASS\n\n";


    // --------------------------------------------------------
    // OUTPUT
    // --------------------------------------------------------

    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report
                << "FAIL: VAE output not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Output shape: "
            << formatShape(output)
            << "\n";


    report
            << "Output elements: "
            << output->elementSize()
            << "\n\n";


    bool correctOutput =
            output->dimensions() == 4 &&
            output->length(0) == 1 &&
            output->length(1) == 3 &&
            output->length(2) == 512 &&
            output->length(3) == 512;


    if (!correctOutput) {

        report
                << "FAIL: Wrong VAE output shape.\n";

        report
                << "Expected: [1,3,512,512]\n";

        report
                << "Actual: "
                << formatShape(output)
                << "\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Output shape: PASS\n\n";


    // --------------------------------------------------------
    // OUTPUT COPY
    // --------------------------------------------------------

    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE,
            true);


    output->copyToHostTensor(
            &outputHost);


    float* outputData =
            outputHost.host<float>();


    if (outputData != nullptr) {

        float minValue =
                outputData[0];

        float maxValue =
                outputData[0];


        for (int i = 1;
             i < outputHost.elementSize();
             ++i) {

            minValue =
                    std::min(
                            minValue,
                            outputData[i]);

            maxValue =
                    std::max(
                            maxValue,
                            outputData[i]);
        }


        report
                << "Output[0]: "
                << outputData[0]
                << "\n";


        report
                << "Output min: "
                << minValue
                << "\n";


        report
                << "Output max: "
                << maxValue
                << "\n\n";
    }


    report
            << "========================================\n"
            << "VAE FP16 CPU TEST: PASS\n"
            << "========================================\n";


    destroyInterpreter(
            interpreter,
            session);


    return report.str();
}


// ============================================================
// VAE OPENCL TEST
// ============================================================

std::string runVaeOpenClTest(
        const std::string& vaePath) {

    std::ostringstream report;

    report
            << "========================================\n"
            << "SANA VAE OPENCL TEST\n"
            << "========================================\n\n";


    long long fileSize =
            getFileSize(vaePath);


    report
            << "Model file size: "
            << fileSize
            << " bytes\n\n";


    if (fileSize <= 0) {

        report
                << "FAIL: VAE file not found.\n";

        return report.str();
    }


    report
            << "Creating MNN interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    if (interpreter == nullptr) {

        report
                << "FAIL: OpenCL interpreter creation failed.\n";

        return report.str();
    }


    report
            << "Interpreter: PASS\n\n";


    report
            << "Creating OpenCL session...\n";


    auto start =
            std::chrono::steady_clock::now();


    MNN::Session* session =
            createSession(
                    interpreter,
                    true,
                    4);


    auto end =
            std::chrono::steady_clock::now();


    double sessionMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                            end - start)
            .count();


    report
            << "OpenCL session creation time: "
            << sessionMs
            << " ms\n";


    if (session == nullptr) {

        report
                << "FAIL: OpenCL session creation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "OpenCL session: PASS\n\n";


    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        input =
                interpreter->getSessionInput(
                        session,
                        nullptr);
    }


    if (input == nullptr) {

        report
                << "FAIL: OpenCL VAE input not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Input shape: "
            << formatShape(input)
            << "\n";


    bool correctShape =
            input->dimensions() == 4 &&
            input->length(0) == 1 &&
            input->length(1) == 32 &&
            input->length(2) == 16 &&
            input->length(3) == 16;


    if (!correctShape) {

        report
                << "FAIL: OpenCL input shape incorrect.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE,
            true);


    float* data =
            hostTensor.host<float>();


    if (data == nullptr) {

        report
                << "FAIL: OpenCL host tensor allocation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    int elements =
            hostTensor.elementSize();


    for (int i = 0;
         i < elements;
         ++i) {

        data[i] =
                -0.05f +
                (
                    static_cast<float>(
                            i % 256)
                    / 255.0f
                ) * 0.10f;
    }


    if (!input->copyFromHostTensor(
                &hostTensor)) {

        report
                << "FAIL: OpenCL input copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Host -> OpenCL: PASS\n\n";


    report
            << "Running OpenCL VAE inference...\n";


    auto startInference =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto endInference =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                            endInference -
                            startInference)
            .count();


    report
            << "Inference time: "
            << inferenceMs
            << " ms\n";


    report
            << "MNN error code: "
            << static_cast<int>(
                    result)
            << "\n\n";


    if (result != MNN::NO_ERROR) {

        report
                << "FAIL: OpenCL VAE inference failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report
                << "FAIL: OpenCL output not found.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Output shape: "
            << formatShape(output)
            << "\n";


    bool correctOutput =
            output->dimensions() == 4 &&
            output->length(0) == 1 &&
            output->length(1) == 3 &&
            output->length(2) == 512 &&
            output->length(3) == 512;


    if (!correctOutput) {

        report
                << "FAIL: OpenCL output shape incorrect.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Output shape: PASS\n\n";


    report
            << "========================================\n"
            << "VAE OPENCL TEST: PASS\n"
            << "========================================\n";


    destroyInterpreter(
            interpreter,
            session);


    return report.str();
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


    long long fileSize =
            getFileSize(transformerPath);


    report
            << "Transformer file size: "
            << fileSize
            << " bytes\n\n";


    if (fileSize <= 0) {

        report
                << "FAIL: Transformer file not found.\n";

        return report.str();
    }


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    transformerPath.c_str());


    if (interpreter == nullptr) {

        report
                << "FAIL: Transformer interpreter creation failed.\n";

        return report.str();
    }


    report
            << "Transformer interpreter: PASS\n";


    MNN::Session* session =
            createSession(
                    interpreter,
                    preferOpenCl,
                    4);


    if (session == nullptr) {

        report
                << "FAIL: Transformer session creation failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Transformer session: PASS\n\n";


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
                << "FAIL: Required Transformer inputs missing.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "encoder_hidden_states: "
            << formatShape(encoderInput)
            << "\n";


    report
            << "hidden_states: "
            << formatShape(hiddenInput)
            << "\n";


    report
            << "timestep: "
            << formatShape(timestepInput)
            << "\n\n";


    MNN::Tensor encoderHost(
            encoderInput,
            MNN::Tensor::CAFFE,
            true);


    MNN::Tensor hiddenHost(
            hiddenInput,
            MNN::Tensor::CAFFE,
            true);


    MNN::Tensor timestepHost(
            timestepInput,
            MNN::Tensor::CAFFE,
            true);


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
                << "FAIL: Transformer host tensor allocation failed.\n";

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


    std::fill(
            timestepData,
            timestepData +
            timestepHost.elementSize(),
            0.0f);


    if (!encoderInput->copyFromHostTensor(
                &encoderHost)) {

        report
                << "FAIL: Encoder copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (!hiddenInput->copyFromHostTensor(
                &hiddenHost)) {

        report
                << "FAIL: Hidden-state copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    if (!timestepInput->copyFromHostTensor(
                &timestepHost)) {

        report
                << "FAIL: Timestep copy failed.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Inputs: PASS\n\n";


    report
            << "Running Transformer...\n";


    auto start =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto end =
            std::chrono::steady_clock::now();


    double elapsed =
            std::chrono::duration<
                    double,
                    std::milli>(
                            end - start)
            .count();


    report
            << "Time: "
            << elapsed
            << " ms\n";


    report
            << "Error code: "
            << static_cast<int>(
                    result)
            << "\n\n";


    if (result != MNN::NO_ERROR) {

        report
                << "FAIL: Transformer inference failed.\n";

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
                << "FAIL: Transformer output missing.\n";

        destroyInterpreter(
                interpreter,
                session);

        return report.str();
    }


    report
            << "Output shape: "
            << formatShape(output)
            << "\n";


    report
            << "Output elements: "
            << output->elementSize()
            << "\n\n";


    report
            << "PASS: Transformer executed successfully.\n";


    destroyInterpreter(
            interpreter,
            session);


    return report.str();
}


// ============================================================
// JNI INITIALIZE
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


    // --------------------------------------------------------
    // IMPORTANT:
    // Always release any previous model first.
    // --------------------------------------------------------

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
    // COPY MODEL ASSET TO CACHE
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

    gStatus =
            "Creating Sana interpreter";


    gInterpreter =
            MNN::Interpreter::createFromFile(
                    modelPath.c_str());


    if (gInterpreter == nullptr) {

        gStatus =
                "Could not create MNN interpreter";

        return JNI_FALSE;
    }


    // --------------------------------------------------------
    // IMPORTANT:
    // Do NOT force OpenCL during startup.
    //
    // The model initializer now uses CPU.
    // OpenCL can be tested separately.
    // --------------------------------------------------------

    gStatus =
            "Creating Sana CPU session";


    int threads =
            static_cast<int>(
                    cpuThreads);


    threads =
            std::max(
                    1,
                    std::min(
                            threads,
                            8));


    gSession =
            createSession(
                    gInterpreter,
                    false,
                    threads);


    if (gSession == nullptr) {

        delete gInterpreter;

        gInterpreter = nullptr;

        gStatus =
                "Could not create Sana CPU session";

        return JNI_FALSE;
    }


    gInitialized = true;

    gBackend =
            "CPU";


    gStatus =
            "Sana model initialized";


    LOGI(
            "Sana initialized using CPU backend");


    (void)preferOpenCl;


    return JNI_TRUE;
}


// ============================================================
// JNI STATE
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


    std::string result =
            runTransformerDiagnostic(
                    transformer,
                    cache,
                    preferOpenCl);


    return env->NewStringUTF(
            result.c_str());
}


// ============================================================
// JNI VAE TEST
//
// IMPORTANT:
// This function now tests CPU first only.
//
// It does NOT automatically run OpenCL.
// This prevents the APK from getting stuck because of
// OpenCL/FP16 session initialization.
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
            runVaeCpuTest(
                    vae);


    return env->NewStringUTF(
            result.c_str());
}


// ============================================================
// JNI BOTH MODELS
//
// Transformer + VAE CPU diagnostic.
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
            << runVaeCpuTest(
                    vae)
            << "\n";


    return env->NewStringUTF(
            report.str().c_str());
}

} // namespace
