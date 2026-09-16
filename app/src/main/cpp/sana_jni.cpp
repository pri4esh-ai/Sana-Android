#include <jni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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
std::string gBackend = "NONE";
std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;
MNN::Session* gSession = nullptr;


// ------------------------------------------------------------
// Helpers
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


std::string formatShape(const std::vector<int>& shape) {
    std::ostringstream out;

    out << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }

        out << shape[i];
    }

    out << "]";

    return out.str();
}


long long getFileSize(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file.good()) {
        return -1;
    }

    return static_cast<long long>(file.tellg());
}


bool isFiniteFloat(float value) {
    return std::isfinite(value);
}


// ------------------------------------------------------------
// Session configuration
// ------------------------------------------------------------

MNN::ScheduleConfig makeCpuScheduleConfig(int cpuThreads = 4) {

    MNN::ScheduleConfig config;

    config.type = MNN_FORWARD_CPU;
    config.numThread = cpuThreads;

    return config;
}


MNN::ScheduleConfig makeOpenClScheduleConfig() {

    MNN::ScheduleConfig config;

    config.type = MNN_FORWARD_OPENCL;
    config.numThread = 4;

    config.mode = MNN_GPU_TUNING_HEAVY;

    // If an OpenCL operation is unsupported,
    // allow MNN to fall back to CPU.
    config.backupType = MNN_FORWARD_CPU;

    return config;
}


// ------------------------------------------------------------
// Cleanup
// ------------------------------------------------------------

void releaseGlobalSession() {

    if (gInterpreter != nullptr && gSession != nullptr) {
        gInterpreter->releaseSession(gSession);
        gSession = nullptr;
    }

    if (gInterpreter != nullptr) {
        delete gInterpreter;
        gInterpreter = nullptr;
    }

    gInitialized = false;
    gBackend = "NONE";
}


// ------------------------------------------------------------
// Create CPU session
// ------------------------------------------------------------

MNN::Session* createCpuSession(
        MNN::Interpreter* interpreter,
        int cpuThreads = 4) {

    if (interpreter == nullptr) {
        return nullptr;
    }

    MNN::ScheduleConfig config =
            makeCpuScheduleConfig(cpuThreads);

    return interpreter->createSession(config);
}


// ------------------------------------------------------------
// Create OpenCL session
// ------------------------------------------------------------

MNN::Session* createOpenClSession(
        MNN::Interpreter* interpreter) {

    if (interpreter == nullptr) {
        return nullptr;
    }

    MNN::ScheduleConfig config =
            makeOpenClScheduleConfig();

    return interpreter->createSession(config);
}


// ------------------------------------------------------------
// Transformer diagnostic
// ------------------------------------------------------------

std::string runTransformerCpuTest(
        const std::string& transformerPath) {

    std::ostringstream report;

    report << "============================================\n";
    report << "SANA TRANSFORMER CPU TEST\n";
    report << "============================================\n";

    report << "Model:\n";
    report << transformerPath << "\n";

    long long fileSize = getFileSize(transformerPath);

    if (fileSize <= 0) {

        report << "File: FAIL\n";
        report << "File does not exist or is empty.\n";

        return report.str();
    }

    report << "File size: "
           << fileSize
           << " bytes\n\n";


    auto createStart =
            std::chrono::steady_clock::now();

    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    transformerPath.c_str());

    auto createEnd =
            std::chrono::steady_clock::now();

    double createMs =
            std::chrono::duration<double, std::milli>(
                    createEnd - createStart).count();

    report << "Creating MNN interpreter...\n";

    report << "Interpreter creation time: "
           << createMs
           << " ms\n";


    if (interpreter == nullptr) {

        report << "Interpreter: FAIL\n";

        return report.str();
    }

    report << "Interpreter: PASS\n";


    auto sessionStart =
            std::chrono::steady_clock::now();

    MNN::Session* session =
            createCpuSession(interpreter, 4);

    auto sessionEnd =
            std::chrono::steady_clock::now();

    double sessionMs =
            std::chrono::duration<double, std::milli>(
                    sessionEnd - sessionStart).count();

    report << "Creating CPU session...\n";

    report << "Session creation time: "
           << sessionMs
           << " ms\n";


    if (session == nullptr) {

        report << "CPU session: FAIL\n";

        delete interpreter;

        return report.str();
    }

    report << "CPU session: PASS\n";


    MNN::Tensor* hiddenStates =
            interpreter->getSessionInput(
                    session,
                    "hidden_states");

    MNN::Tensor* timestep =
            interpreter->getSessionInput(
                    session,
                    "timestep");

    MNN::Tensor* encoderHiddenStates =
            interpreter->getSessionInput(
                    session,
                    "encoder_hidden_states");


    if (hiddenStates == nullptr ||
        timestep == nullptr ||
        encoderHiddenStates == nullptr) {

        report << "Required transformer inputs: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Transformer inputs: PASS\n";


    // --------------------------------------------------------
    // Hidden states
    // --------------------------------------------------------

    std::vector<int> hiddenShape =
            hiddenStates->shape();

    report << "hidden_states shape: "
           << formatShape(hiddenShape)
           << "\n";


    MNN::Tensor hiddenHost(
            hiddenStates,
            MNN::Tensor::CAFFE);

    float* hiddenPtr =
            hiddenHost.host<float>();

    if (hiddenPtr == nullptr) {

        report << "hidden_states host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }

    std::fill(
            hiddenPtr,
            hiddenPtr + hiddenHost.elementSize(),
            0.0f);

    hiddenStates->copyFromHostTensor(&hiddenHost);


    // --------------------------------------------------------
    // Timestep
    // --------------------------------------------------------

    MNN::Tensor timestepHost(
            timestep,
            MNN::Tensor::CAFFE);

    float* timestepPtr =
            timestepHost.host<float>();

    if (timestepPtr == nullptr) {

        report << "timestep host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }

    std::fill(
            timestepPtr,
            timestepPtr + timestepHost.elementSize(),
            0.0f);

    timestep->copyFromHostTensor(&timestepHost);


    // --------------------------------------------------------
    // Encoder hidden states
    // --------------------------------------------------------

    MNN::Tensor encoderHost(
            encoderHiddenStates,
            MNN::Tensor::CAFFE);

    float* encoderPtr =
            encoderHost.host<float>();

    if (encoderPtr == nullptr) {

        report << "encoder_hidden_states host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }

    std::fill(
            encoderPtr,
            encoderPtr + encoderHost.elementSize(),
            0.0f);

    encoderHiddenStates->copyFromHostTensor(
            &encoderHost);


    report << "Input tensors: PASS\n";


    auto inferenceStart =
            std::chrono::steady_clock::now();

    MNN::ErrorCode result =
            interpreter->runSession(session);

    auto inferenceEnd =
            std::chrono::steady_clock::now();

    double inferenceMs =
            std::chrono::duration<double, std::milli>(
                    inferenceEnd - inferenceStart).count();


    report << "Running transformer inference...\n";

    report << "Inference time: "
           << inferenceMs
           << " ms\n";

    report << "MNN error code: "
           << static_cast<int>(result)
           << "\n";


    if (result != MNN::NO_ERROR) {

        report << "Inference: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }

    report << "Inference: PASS\n";


    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    "sample");


    if (output == nullptr) {

        report << "Output tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Output shape: "
           << formatShape(output->shape())
           << "\n";


    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);

    output->copyToHostTensor(&outputHost);

    float* outputPtr =
            outputHost.host<float>();


    if (outputPtr != nullptr &&
        outputHost.elementSize() > 0) {

        report << "Output[0]: "
               << outputPtr[0]
               << "\n";

        bool finite = true;

        for (size_t i = 0;
             i < outputHost.elementSize();
             ++i) {

            if (!isFiniteFloat(outputPtr[i])) {
                finite = false;
                break;
            }
        }

        report << "Output finite: "
               << (finite ? "YES" : "NO")
               << "\n";
    }


    interpreter->releaseSession(session);
    delete interpreter;


    report << "============================================\n";
    report << "SANA TRANSFORMER CPU TEST: PASS\n";
    report << "============================================\n";


    return report.str();
}


// ------------------------------------------------------------
// VAE CPU diagnostic
//
// IMPORTANT:
// This version deliberately uses an ALL-ZERO latent.
// No random/sequential latent values are used.
// ------------------------------------------------------------

std::string runVaeCpuTest(
        const std::string& vaePath) {

    std::ostringstream report;


    report << "============================================\n";
    report << "SANA VAE FP16 CPU TEST\n";
    report << "============================================\n";


    report << "Model:\n";
    report << vaePath\n";
    report << "\n";


    long long fileSize =
            getFileSize(vaePath);


    if (fileSize <= 0) {

        report << "File: FAIL\n";
        report << "File does not exist or is empty.\n";

        return report.str();
    }


    report << "File size: "
           << fileSize
           << " bytes\n\n";


    // --------------------------------------------------------
    // Interpreter
    // --------------------------------------------------------

    auto createStart =
            std::chrono::steady_clock::now();


    report << "Creating MNN interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    auto createEnd =
            std::chrono::steady_clock::now();


    double createMs =
            std::chrono::duration<double, std::milli>(
                    createEnd - createStart).count();


    report << "Interpreter creation time: "
           << createMs
           << " ms\n";


    if (interpreter == nullptr) {

        report << "Interpreter: FAIL\n";

        return report.str();
    }


    report << "Interpreter: PASS\n";


    // --------------------------------------------------------
    // CPU session
    // --------------------------------------------------------

    report << "Creating CPU session...\n";


    auto sessionStart =
            std::chrono::steady_clock::now();


    MNN::Session* session =
            createCpuSession(
                    interpreter,
                    4);


    auto sessionEnd =
            std::chrono::steady_clock::now();


    double sessionMs =
            std::chrono::duration<double, std::milli>(
                    sessionEnd - sessionStart).count();


    report << "Session creation time: "
           << sessionMs
           << " ms\n";


    if (session == nullptr) {

        report << "CPU session: FAIL\n";

        delete interpreter;

        return report.str();
    }


    report << "CPU session: PASS\n";


    // --------------------------------------------------------
    // Input
    // --------------------------------------------------------

    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        report << "Input tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    std::vector<int> inputShape =
            input->shape();


    report << "Input shape: "
           << formatShape(inputShape)
           << "\n";


    size_t inputElements =
            input->elementSize();


    report << "Input elements: "
           << inputElements
           << "\n";


    report << "Input type bytes: "
           << input->getType().bytes()
           << "\n";


    report << "Input format: CAFFE / NCHW\n";


    // --------------------------------------------------------
    // Verify expected Sana VAE latent shape
    // --------------------------------------------------------

    bool correctShape =
            inputShape.size() == 4 &&
            inputShape[0] == 1 &&
            inputShape[1] == 32 &&
            inputShape[2] == 16 &&
            inputShape[3] == 16;


    if (!correctShape) {

        report << "Input shape: FAIL\n";
        report << "Expected [1, 32, 16, 16]\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Input shape: PASS\n";


    // --------------------------------------------------------
    // Create host tensor
    // --------------------------------------------------------

    report << "Creating host tensor...\n";


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    float* hostPtr =
            hostTensor.host<float>();


    if (hostPtr == nullptr) {

        report << "Host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Host tensor: PASS\n";


    // --------------------------------------------------------
    // IMPORTANT:
    //
    // ZERO LATENT TEST
    //
    // Every single latent value is exactly 0.0f.
    // --------------------------------------------------------

    std::fill(
            hostPtr,
            hostPtr + hostTensor.elementSize(),
            0.0f);


    report << "ZERO LATENT TEST\n";
    report << "All latent values set to exactly 0.0\n";


    report << "Latent sample: "
           << hostPtr[0]
           << ", "
           << hostPtr[1]
           << ", "
           << hostPtr[2]
           << "\n";


    report << "Latent zero verification: ";

    bool allZero = true;

    for (size_t i = 0;
         i < hostTensor.elementSize();
         ++i) {

        if (hostPtr[i] != 0.0f) {
            allZero = false;
            break;
        }
    }


    report << (allZero ? "PASS" : "FAIL")
           << "\n";


    if (!allZero) {

        report << "Zero latent preparation failed.\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    // --------------------------------------------------------
    // Copy to MNN
    // --------------------------------------------------------

    report << "Copying ZERO latent to MNN...\n";


    input->copyFromHostTensor(
            &hostTensor);


    report << "Host -> MNN: PASS\n";


    // --------------------------------------------------------
    // Run VAE
    // --------------------------------------------------------

    report << "Running VAE inference...\n";


    auto inferenceStart =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(session);


    auto inferenceEnd =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<double, std::milli>(
                    inferenceEnd - inferenceStart).count();


    report << "Inference time: "
           << inferenceMs
           << " ms\n";


    report << "MNN error code: "
           << static_cast<int>(result)
           << "\n";


    if (result != MNN::NO_ERROR) {

        report << "Inference: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Inference: PASS\n";


    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report << "Output tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    std::vector<int> outputShape =
            output->shape();


    report << "Output shape: "
           << formatShape(outputShape)
           << "\n";


    size_t outputElements =
            output->elementSize();


    report << "Output elements: "
           << outputElements
           << "\n";


    bool correctOutputShape =
            outputShape.size() == 4 &&
            outputShape[0] == 1 &&
            outputShape[1] == 3 &&
            outputShape[2] == 512 &&
            outputShape[3] == 512;


    if (!correctOutputShape) {

        report << "Output shape: FAIL\n";
        report << "Expected [1, 3, 512, 512]\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Output shape: PASS\n";


    // --------------------------------------------------------
    // Copy output to host
    // --------------------------------------------------------

    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);


    output->copyToHostTensor(
            &outputHost);


    float* outputPtr =
            outputHost.host<float>();


    if (outputPtr == nullptr) {

        report << "Output host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    // --------------------------------------------------------
    // Check output values
    // --------------------------------------------------------

    float minimum =
            std::numeric_limits<float>::infinity();

    float maximum =
            -std::numeric_limits<float>::infinity();

    bool finite =
            true;


    for (size_t i = 0;
         i < outputHost.elementSize();
         ++i) {

        float value =
                outputPtr[i];


        if (!std::isfinite(value)) {

            finite = false;

            break;
        }


        minimum =
                std::min(
                        minimum,
                        value);


        maximum =
                std::max(
                        maximum,
                        value);
    }


    report << "Output[0]: "
           << outputPtr[0]
           << "\n";


    if (finite) {

        report << "Output min: "
               << minimum
               << "\n";


        report << "Output max: "
               << maximum
               << "\n";


        report << "Output finite: YES\n";

    } else {

        report << "Output min: nan\n";
        report << "Output max: nan\n";
        report << "Output finite: NO\n";
    }


    // --------------------------------------------------------
    // Final diagnostic result
    // --------------------------------------------------------

    interpreter->releaseSession(session);
    delete interpreter;


    report << "============================================\n";


    if (finite) {

        report << "ZERO LATENT VAE TEST: PASS\n";

    } else {

        report << "ZERO LATENT VAE TEST: FAILED NUMERIC OUTPUT\n";
    }


    report << "============================================\n";


    return report.str();
}


// ------------------------------------------------------------
// VAE OpenCL diagnostic
// ------------------------------------------------------------

std::string runVaeOpenClTest(
        const std::string& vaePath) {

    std::ostringstream report;


    report << "============================================\n";
    report << "SANA VAE OPENCL TEST\n";
    report << "============================================\n";


    report << "Model:\n";
    report << vaePath
           << "\n";


    long long fileSize =
            getFileSize(vaePath);


    if (fileSize <= 0) {

        report << "File: FAIL\n";

        return report.str();
    }


    report << "File size: "
           << fileSize
           << " bytes\n";


    report << "Creating MNN interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    if (interpreter == nullptr) {

        report << "Interpreter: FAIL\n";

        return report.str();
    }


    report << "Interpreter: PASS\n";


    report << "Creating OpenCL session...\n";


    MNN::Session* session =
            createOpenClSession(
                    interpreter);


    if (session == nullptr) {

        report << "OpenCL session: FAIL\n";

        delete interpreter;

        return report.str();
    }


    report << "OpenCL session: PASS\n";


    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        report << "Input tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    std::vector<int> inputShape =
            input->shape();


    report << "Input shape: "
           << formatShape(inputShape)
           << "\n";


    bool correctShape =
            inputShape.size() == 4 &&
            inputShape[0] == 1 &&
            inputShape[1] == 32 &&
            inputShape[2] == 16 &&
            inputShape[3] == 16;


    if (!correctShape) {

        report << "Input shape: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Input shape: PASS\n";


    // --------------------------------------------------------
    // ZERO LATENT
    // --------------------------------------------------------

    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    float* hostPtr =
            hostTensor.host<float>();


    if (hostPtr == nullptr) {

        report << "Host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    std::fill(
            hostPtr,
            hostPtr + hostTensor.elementSize(),
            0.0f);


    report << "ZERO LATENT TEST\n";
    report << "All latent values set to exactly 0.0\n";


    report << "Latent sample: "
           << hostPtr[0]
           << ", "
           << hostPtr[1]
           << ", "
           << hostPtr[2]
           << "\n";


    input->copyFromHostTensor(
            &hostTensor);


    report << "Host -> MNN: PASS\n";


    report << "Running VAE inference...\n";


    auto inferenceStart =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(session);


    auto inferenceEnd =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<double, std::milli>(
                    inferenceEnd - inferenceStart).count();


    report << "Inference time: "
           << inferenceMs
           << " ms\n";


    report << "MNN error code: "
           << static_cast<int>(result)
           << "\n";


    if (result != MNN::NO_ERROR) {

        report << "Inference: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Inference: PASS\n";


    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report << "Output tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    std::vector<int> outputShape =
            output->shape();


    report << "Output shape: "
           << formatShape(outputShape)
           << "\n";


    bool correctOutputShape =
            outputShape.size() == 4 &&
            outputShape[0] == 1 &&
            outputShape[1] == 3 &&
            outputShape[2] == 512 &&
            outputShape[3] == 512;


    if (!correctOutputShape) {

        report << "Output shape: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    report << "Output shape: PASS\n";


    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);


    output->copyToHostTensor(
            &outputHost);


    float* outputPtr =
            outputHost.host<float>();


    if (outputPtr == nullptr) {

        report << "Output host tensor: FAIL\n";

        interpreter->releaseSession(session);
        delete interpreter;

        return report.str();
    }


    float minimum =
            std::numeric_limits<float>::infinity();

    float maximum =
            -std::numeric_limits<float>::infinity();

    bool finite =
            true;


    for (size_t i = 0;
         i < outputHost.elementSize();
         ++i) {

        float value =
                outputPtr[i];


        if (!std::isfinite(value)) {

            finite = false;
            break;
        }


        minimum =
                std::min(
                        minimum,
                        value);


        maximum =
                std::max(
                        maximum,
                        value);
    }


    report << "Output[0]: "
           << outputPtr[0]
           << "\n";


    if (finite) {

        report << "Output min: "
               << minimum
               << "\n";


        report << "Output max: "
               << maximum
               << "\n";


        report << "Output finite: YES\n";

    } else {

        report << "Output min: nan\n";
        report << "Output max: nan\n";
        report << "Output finite: NO\n";
    }


    interpreter->releaseSession(session);
    delete interpreter;


    report << "============================================\n";


    if (finite) {

        report << "ZERO LATENT VAE OPENCL TEST: PASS\n";

    } else {

        report << "ZERO LATENT VAE OPENCL TEST: FAILED NUMERIC OUTPUT\n";
    }


    report << "============================================\n";


    return report.str();
}

} // namespace


// ============================================================
// JNI
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv* env,
        jobject /* thiz */,
        jobject assetManager,
        jstring modelAsset,
        jstring cachePath,
        jboolean /* preferOpenCl */,
        jint cpuThreads) {

    std::lock_guard<std::mutex> lock(gMutex);


    releaseGlobalSession();


    if (assetManager == nullptr ||
        modelAsset == nullptr ||
        cachePath == nullptr) {

        gStatus =
                "Invalid initialization arguments";

        return JNI_FALSE;
    }


    std::string assetName =
            jstringToString(
                    env,
                    modelAsset);


    std::string cacheDirectory =
            jstringToString(
                    env,
                    cachePath);


    if (assetName.empty() ||
        cacheDirectory.empty()) {

        gStatus =
                "Invalid model asset or cache path";

        return JNI_FALSE;
    }


    AAssetManager* manager =
            AAssetManager_fromJava(
                    env,
                    assetManager);


    if (manager == nullptr) {

        gStatus =
                "Unable to access AssetManager";

        return JNI_FALSE;
    }


    AAsset* asset =
            AAssetManager_open(
                    manager,
                    assetName.c_str(),
                    AASSET_MODE_STREAMING);


    if (asset == nullptr) {

        gStatus =
                "Unable to open model asset: "
                + assetName;

        return JNI_FALSE;
    }


    std::string destination =
            cacheDirectory +
            "/sana_model.mnn";


    FILE* outputFile =
            fopen(
                    destination.c_str(),
                    "wb");


    if (outputFile == nullptr) {

        AAsset_close(asset);

        gStatus =
                "Unable to create cached model";

        return JNI_FALSE;
    }


    char buffer[1024 * 1024];


    int bytesRead = 0;


    while ((bytesRead =
            AAsset_read(
                    asset,
                    buffer,
                    sizeof(buffer))) > 0) {

        fwrite(
                buffer,
                1,
                bytesRead,
                outputFile);
    }


    fclose(outputFile);
    AAsset_close(asset);


    gStatus =
            "Creating Sana CPU interpreter";


    gInterpreter =
            MNN::Interpreter::createFromFile(
                    destination.c_str());


    if (gInterpreter == nullptr) {

        gStatus =
                "Failed to create Sana interpreter";

        releaseGlobalSession();

        return JNI_FALSE;
    }


    int threads =
            std::max(
                    1,
                    std::min(
                            8,
                            static_cast<int>(
                                    cpuThreads)));


    gSession =
            createCpuSession(
                    gInterpreter,
                    threads);


    if (gSession == nullptr) {

        gStatus =
                "Failed to create Sana CPU session";

        releaseGlobalSession();

        return JNI_FALSE;
    }


    gInitialized = true;

    gBackend = "CPU";

    gStatus =
            "Sana initialized successfully on CPU";


    return JNI_TRUE;
}


// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv* /* env */,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(gMutex);

    return gInitialized
           ? JNI_TRUE
           : JNI_FALSE;
}


// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(gMutex);

    return env->NewStringUTF(
            gBackend.c_str());
}


// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(gMutex);

    return env->NewStringUTF(
            gStatus.c_str());
}


// ============================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv* /* env */,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(gMutex);

    releaseGlobalSession();

    gBackend = "NONE";

    gStatus = "Released";
}


// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
        JNIEnv* env,
        jobject /* thiz */,
        jstring transformerPath,
        jstring cachePath,
        jboolean /* preferOpenCl */) {

    std::string path =
            jstringToString(
                    env,
                    transformerPath);

    std::string cache =
            jstringToString(
                    env,
                    cachePath);

    (void)cache;


    std::string result =
            runTransformerCpuTest(
                    path);


    return env->NewStringUTF(
            result.c_str());
}


// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject /* thiz */,
        jstring vaePath,
        jstring cachePath,
        jboolean /* preferOpenCl */) {

    std::string path =
            jstringToString(
                    env,
                    vaePath);

    std::string cache =
            jstringToString(
                    env,
                    cachePath);

    (void)cache;


    // --------------------------------------------------------
    // IMPORTANT:
    // VAE diagnostic intentionally uses CPU only.
    //
    // The latent is ALL ZERO.
    // --------------------------------------------------------

    std::string result =
            runVaeCpuTest(
                    path);


    return env->NewStringUTF(
            result.c_str());
}


// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestModels(
        JNIEnv* env,
        jobject /* thiz */,
        jstring transformerPath,
        jstring vaePath,
        jstring cachePath,
        jboolean /* preferOpenCl */) {

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

    (void)cache;


    std::ostringstream report;


    report << "============================================\n";
    report << "SANA MODEL DIAGNOSTIC\n";
    report << "============================================\n\n";


    report << runTransformerCpuTest(
            transformer);


    report << "\n\n";


    report << runVaeCpuTest(
            vae);


    report << "\n============================================\n";


    return env->NewStringUTF(
            report.str().c_str());
}
