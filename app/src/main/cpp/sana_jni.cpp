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


// ============================================================
// JNI STRING
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
// SHAPE
// ============================================================

std::string formatShape(
        const MNN::Tensor* tensor) {

    if (tensor == nullptr) {
        return "null";
    }

    std::vector<int> shape =
        tensor->shape();

    std::ostringstream out;

    out << "[";

    for (size_t i = 0;
         i < shape.size();
         ++i) {

        if (i > 0) {
            out << ", ";
        }

        out << shape[i];
    }

    out << "]";

    return out.str();
}


// ============================================================
// DIMENSION TYPE
// ============================================================

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


// ============================================================
// BACKEND NAME
// ============================================================

std::string backendName(
        bool preferOpenCl) {

    if (preferOpenCl) {
        return "OpenCL / FP16";
    }

    return "CPU";
}


// ============================================================
// FILE SIZE
// ============================================================

long long getFileSize(
        const std::string& path) {

    std::ifstream file(
        path,
        std::ios::binary |
        std::ios::ate
    );

    if (!file.good()) {
        return -1;
    }

    std::streamsize size =
        file.tellg();

    file.close();

    return static_cast<long long>(size);
}


// ============================================================
// DESTROY INTERPRETER
// ============================================================

void destroyInterpreter(
        MNN::Interpreter*& interpreter) {

    if (interpreter != nullptr) {

        MNN::Interpreter::destroy(
            interpreter
        );

        interpreter = nullptr;
    }
}


// ============================================================
// GLOBAL RELEASE
// ============================================================

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
// SESSION CONFIG
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


// ============================================================
// CREATE SESSION
// ============================================================

bool createSession(
        MNN::Interpreter* interpreter,
        MNN::Session*& session,
        bool preferOpenCl,
        int cpuThreads) {

    if (interpreter == nullptr) {
        return false;
    }

    MNN::ScheduleConfig config =
        makeScheduleConfig(
            preferOpenCl,
            cpuThreads
        );

    if (preferOpenCl) {

        config.backupType =
            MNN_FORWARD_CPU;
    }

    session =
        interpreter->createSession(
            config
        );

    if (session == nullptr) {

        LOGE(
            "MNN createSession failed"
        );

        return false;
    }

    return true;
}


// ============================================================
// FILL TEST LATENT
// ============================================================

void fillTestLatent(
        MNN::Tensor* tensor) {

    if (tensor == nullptr) {
        return;
    }

    float* data =
        tensor->host<float>();

    if (data == nullptr) {
        return;
    }

    const int elements =
        tensor->elementSize();

    for (int i = 0;
         i < elements;
         ++i) {

        data[i] =
            -0.05f +
            0.000390625f *
            static_cast<float>(
                i % 256
            );
    }
}


// ============================================================
// TRANSFORMER DIAGNOSTIC
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


    // --------------------------------------------------------
    // INTERPRETER
    // --------------------------------------------------------

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
            << "FAIL: Could not create Transformer session.";

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Backend: "
        << backendName(preferOpenCl)
        << "\n\n";


    // --------------------------------------------------------
    // INPUTS
    // --------------------------------------------------------

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
            << "FAIL: Transformer inputs not found.\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Actual backend: "
        << backendName(preferOpenCl)
        << "\n\n";


    result
        << "Inputs: 3\n\n";


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


    // --------------------------------------------------------
    // HOST TENSORS
    // --------------------------------------------------------

    MNN::Tensor* encoderHost =
        new MNN::Tensor(
            encoderInput,
            encoderInput->getDimensionType()
        );


    MNN::Tensor* hiddenHost =
        new MNN::Tensor(
            hiddenInput,
            hiddenInput->getDimensionType()
        );


    MNN::Tensor* timestepHost =
        new MNN::Tensor(
            timestepInput,
            timestepInput->getDimensionType()
        );


    float* encoderData =
        encoderHost->host<float>();


    float* hiddenData =
        hiddenHost->host<float>();


    float* timestepData =
        timestepHost->host<float>();


    if (encoderData == nullptr ||
        hiddenData == nullptr ||
        timestepData == nullptr) {

        result
            << "FAIL: Could not allocate Transformer host tensors.\n";

        delete encoderHost;
        delete hiddenHost;
        delete timestepHost;

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    // --------------------------------------------------------
    // TEST DATA
    // --------------------------------------------------------

    for (int i = 0;
         i < encoderHost->elementSize();
         ++i) {

        encoderData[i] =
            0.001f *
            static_cast<float>(
                (i % 100) - 50
            );
    }


    for (int i = 0;
         i < hiddenHost->elementSize();
         ++i) {

        hiddenData[i] =
            0.001f *
            static_cast<float>(
                (i % 100) - 50
            );
    }


    timestepData[0] =
        0.5f;


    // --------------------------------------------------------
    // COPY
    // --------------------------------------------------------

    bool encoderCopied =
        encoderInput->copyFromHostTensor(
            encoderHost
        );


    bool hiddenCopied =
        hiddenInput->copyFromHostTensor(
            hiddenHost
        );


    bool timestepCopied =
        timestepInput->copyFromHostTensor(
            timestepHost
        );


    delete encoderHost;
    delete hiddenHost;
    delete timestepHost;


    if (!encoderCopied ||
        !hiddenCopied ||
        !timestepCopied) {

        result
            << "FAIL: Could not copy Transformer input tensors.\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    // --------------------------------------------------------
    // RUN
    // --------------------------------------------------------

    result
        << "Running inference...\n\n";


    auto start =
        std::chrono::steady_clock::now();


    int errorCode =
        interpreter->runSession(
            session
        );


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


    // --------------------------------------------------------
    // OUTPUT
    // --------------------------------------------------------

    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            "sample"
        );


    if (output == nullptr) {

        result
            << "FAIL: Transformer output not found.\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

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
            << "PASS: Transformer executed successfully\n";

    } else {

        result
            << "FAIL: Transformer inference error\n";
    }


    interpreter->releaseSession(
        session
    );


    destroyInterpreter(
        interpreter
    );


    result
        << "\nTransformer released successfully.";


    return result.str();
}


// ============================================================
// VAE STAGING-TENSOR TEST
// ============================================================

std::string runVaeStagingTensorTest(
        const std::string& modelPath,
        bool preferOpenCl) {

    std::ostringstream result;

    result
        << "\n\n"
        << "========================================\n"
        << "VAE STAGING TENSOR TEST\n"
        << "========================================\n\n";


    result
        << "This test uses MNN's device-to-host\n"
        << "staging tensor mechanism.\n\n";


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

        destroyInterpreter(
            interpreter
        );

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
            << "FAIL: VAE latent input not found.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Input shape: "
        << formatShape(input)
        << "\n";


    result
        << "Input elements: "
        << input->elementSize()
        << "\n";


    result
        << "Input type code: "
        << static_cast<int>(
            input->getType().code
        )
        << "\n";


    result
        << "Input bytes: "
        << static_cast<int>(
            input->getType().bytes()
        )
        << "\n";


    result
        << "Input dimension type: "
        << dimensionTypeName(
            input->getDimensionType()
        )
        << "\n";


    result
        << "Input device ID: "
        << input->deviceId()
        << "\n\n";


    // --------------------------------------------------------
    // RESIZE
    // --------------------------------------------------------

    std::vector<int> expectedShape = {
        1, 32, 16, 16
    };


    result
        << "Ensuring VAE input shape is [1,32,16,16]...\n";


    if (input->shape() != expectedShape) {

        interpreter->resizeTensor(
            input,
            expectedShape
        );

        interpreter->resizeSession(
            session
        );
    }


    result
        << "Final input shape: "
        << formatShape(input)
        << "\n\n";


    // --------------------------------------------------------
    // CREATE HOST STAGING TENSOR
    // --------------------------------------------------------

    result
        << "Creating host tensor from device tensor...\n";


    MNN::Tensor* hostTensor =
        MNN::Tensor::createHostTensorFromDevice(
            input,
            false
        );


    if (hostTensor == nullptr) {

        result
            << "FAIL: createHostTensorFromDevice(false) returned null.\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "PASS: Host staging tensor created.\n\n";


    // --------------------------------------------------------
    // HOST INFORMATION
    // --------------------------------------------------------

    result
        << "Host staging shape: "
        << formatShape(hostTensor)
        << "\n";


    result
        << "Host staging elements: "
        << hostTensor->elementSize()
        << "\n";


    result
        << "Host staging type code: "
        << static_cast<int>(
            hostTensor->getType().code
        )
        << "\n";


    result
        << "Host staging bytes: "
        << static_cast<int>(
            hostTensor->getType().bytes()
        )
        << "\n";


    result
        << "Host staging dimension type: "
        << dimensionTypeName(
            hostTensor->getDimensionType()
        )
        << "\n";


    result
        << "Host staging device ID: "
        << hostTensor->deviceId()
        << "\n\n";


    // --------------------------------------------------------
    // HOST POINTER
    // --------------------------------------------------------

    result
        << "Getting host staging pointer...\n";


    float* hostData =
        hostTensor->host<float>();


    if (hostData == nullptr) {

        result
            << "FAIL: Host staging tensor has null host pointer.\n";

        MNN::Tensor::destroy(
            hostTensor
        );

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "PASS: Host staging pointer acquired.\n";


    // --------------------------------------------------------
    // FILL HOST TENSOR
    // --------------------------------------------------------

    fillTestLatent(
        hostTensor
    );


    result
        << "Host latent sample: "
        << hostData[0]
        << ", "
        << hostData[1]
        << ", "
        << hostData[2]
        << ", "
        << hostData[3]
        << "\n\n";


    // --------------------------------------------------------
    // COPY HOST -> DEVICE
    // --------------------------------------------------------

    result
        << "Calling copyFromHostTensor()...\n";


    bool copied =
        input->copyFromHostTensor(
            hostTensor
        );


    result
        << "copyFromHostTensor result: "
        << (copied ? "SUCCESS" : "FAILED")
        << "\n\n";


    if (!copied) {

        result
            << "FAIL: MNN rejected host-to-device staging copy.\n";

        MNN::Tensor::destroy(
            hostTensor
        );

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "PASS: Host tensor copied to device.\n\n";


    // --------------------------------------------------------
    // RUN INFERENCE
    // --------------------------------------------------------

    result
        << "Running VAE inference...\n";


    auto start =
        std::chrono::steady_clock::now();


    int errorCode =
        interpreter->runSession(
            session
        );


    auto end =
        std::chrono::steady_clock::now();


    double elapsed =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();


    result
        << "VAE inference time: "
        << elapsed
        << " ms\n";


    result
        << "VAE error code: "
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
            << "FAIL: VAE output tensor is null.";

        MNN::Tensor::destroy(
            hostTensor
        );

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

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


    result
        << "Output type code: "
        << static_cast<int>(
            output->getType().code
        )
        << "\n";


    result
        << "Output dimension type: "
        << dimensionTypeName(
            output->getDimensionType()
        )
        << "\n\n";


    // --------------------------------------------------------
    // CREATE OUTPUT HOST STAGING TENSOR
    // --------------------------------------------------------

    result
        << "Creating output host staging tensor...\n";


    MNN::Tensor* outputHost =
        MNN::Tensor::createHostTensorFromDevice(
            output,
            false
        );


    if (outputHost == nullptr) {

        result
            << "FAIL: Could not create output host tensor.";

        MNN::Tensor::destroy(
            hostTensor
        );

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Output host staging tensor created.\n";


    result
        << "Copying device output to host...\n";


    bool outputCopied =
        output->copyToHostTensor(
            outputHost
        );


    result
        << "copyToHostTensor result: "
        << (outputCopied
            ? "SUCCESS"
            : "FAILED")
        << "\n";


    if (outputCopied) {

        float* outputData =
            outputHost->host<float>();


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
                << "\n\n";
        }
    }


    // --------------------------------------------------------
    // FINAL RESULT
    // --------------------------------------------------------

    if (errorCode == 0) {

        result
            << "PASS: VAE inference executed successfully.\n";

    } else {

        result
            << "FAIL: VAE inference returned error code "
            << errorCode
            << ".\n";
    }


    if (!outputCopied) {

        result
            << "WARNING: Inference completed but output transfer failed.\n";
    }


    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------

    MNN::Tensor::destroy(
        outputHost
    );


    MNN::Tensor::destroy(
        hostTensor
    );


    interpreter->releaseSession(
        session
    );


    destroyInterpreter(
        interpreter
    );


    result
        << "\nVAE staging-tensor test finished.";


    return result.str();
}


// ============================================================
// VAE OPENCL MAP DIAGNOSTIC
// ============================================================

std::string runVaeOpenClMapDiagnostic(
        const std::string& modelPath) {

    std::ostringstream result;

    result
        << "\n\n"
        << "========================================\n"
        << "VAE OPENCL MAP DIAGNOSTIC\n"
        << "========================================\n\n";


    result
        << "This is a separate diagnostic.\n"
        << "The main VAE test uses the staging tensor path.\n\n";


    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );


    if (interpreter == nullptr) {

        result
            << "FAIL: OpenCL interpreter creation failed.";

        return result.str();
    }


    MNN::Session* session = nullptr;


    if (!createSession(
            interpreter,
            session,
            true,
            4)) {

        result
            << "FAIL: OpenCL session creation failed.";

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


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
            << "FAIL: OpenCL latent input not found.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Input shape: "
        << formatShape(input)
        << "\n";


    result
        << "Input device ID: "
        << input->deviceId()
        << "\n";


    result
        << "Input dimension type: "
        << dimensionTypeName(
            input->getDimensionType()
        )
        << "\n\n";


    int waitResult =
        input->wait(
            MNN::Tensor::MAP_TENSOR_WRITE,
            true
        );


    result
        << "wait(MAP_TENSOR_WRITE,true): "
        << waitResult
        << "\n";


    void* mapped =
        input->map(
            MNN::Tensor::MAP_TENSOR_WRITE,
            input->getDimensionType()
        );


    if (mapped == nullptr) {

        result
            << "MAP RESULT: FAILED\n";

    } else {

        result
            << "MAP RESULT: SUCCESS\n";


        float* data =
            static_cast<float*>(
                mapped
            );


        if (data != nullptr) {

            data[0] = -0.05f;
            data[1] = -0.049609375f;
            data[2] = -0.04921875f;
            data[3] = -0.048828125f;

            result
                << "Mapped pointer is writable.\n";
        }


        input->unmap(
            MNN::Tensor::MAP_TENSOR_WRITE,
            input->getDimensionType(),
            mapped
        );


        result
            << "Unmap completed.\n";
    }


    interpreter->releaseSession(
        session
    );


    destroyInterpreter(
        interpreter
    );


    return result.str();
}


// ============================================================
// VAE DIAGNOSTIC
// ============================================================

std::string runVaeDiagnostic(
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl) {

    (void)cachePath;

    std::ostringstream result;

    result
        << "SANA 0.6B / 512 VAE DECODER TEST\n\n";


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
    // MAIN STAGING TEST
    // --------------------------------------------------------

    result
        << runVaeStagingTensorTest(
            modelPath,
            preferOpenCl
        );


    // --------------------------------------------------------
    // OPENCL MAP TEST
    // --------------------------------------------------------

    if (preferOpenCl) {

        result
            << runVaeOpenClMapDiagnostic(
                modelPath
            );
    }


    return result.str();
}


// ============================================================
// COMBINED DIAGNOSTIC
// ============================================================

std::string runCombinedDiagnostic(
        const std::string& transformerPath,
        const std::string& vaePath,
        const std::string& cachePath,
        bool preferOpenCl) {

    std::ostringstream result;


    result
        << "========================================\n"
        << "SANA 0.6B / 512 MODEL DIAGNOSTIC\n"
        << "========================================\n\n";


    result
        << runTransformerDiagnostic(
            transformerPath,
            cachePath,
            preferOpenCl
        );


    result
        << "\n\n"
        << "========================================\n"
        << "VAE DECODER DIAGNOSTIC\n"
        << "========================================\n";


    result
        << runVaeDiagnostic(
            vaePath,
            cachePath,
            preferOpenCl
        );


    return result.str();
}


// ============================================================
// COPY ASSET TO CACHE
// ============================================================

bool copyAssetToFile(
        AAssetManager* assetManager,
        const std::string& assetName,
        const std::string& destination) {

    if (assetManager == nullptr) {

        LOGE(
            "AssetManager is null"
        );

        return false;
    }


    AAsset* asset =
        AAssetManager_open(
            assetManager,
            assetName.c_str(),
            AASSET_MODE_STREAMING
        );


    if (asset == nullptr) {

        LOGE(
            "Could not open asset: %s",
            assetName.c_str()
        );

        return false;
    }


    FILE* output =
        std::fopen(
            destination.c_str(),
            "wb"
        );


    if (output == nullptr) {

        LOGE(
            "Could not create destination: %s",
            destination.c_str()
        );

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
// JNI INITIALIZE
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


    if (assetName.empty()) {

        gStatus =
            "Model asset path is empty";

        return JNI_FALSE;
    }


    if (cache.empty()) {

        gStatus =
            "Cache path is empty";

        return JNI_FALSE;
    }


    std::string modelPath =
        cache +
        "/sana_model.mnn";


    LOGI(
        "Copying Sana model asset: %s",
        assetName.c_str()
    );


    if (!copyAssetToFile(
            assetManager,
            assetName,
            modelPath
        )) {

        gStatus =
            "Failed to copy Sana model asset";

        return JNI_FALSE;
    }


    long long size =
        getFileSize(modelPath);


    if (size <= 0) {

        gStatus =
            "Copied model file is empty";

        return JNI_FALSE;
    }


    LOGI(
        "Sana model copied. Size=%lld",
        size
    );


    // --------------------------------------------------------
    // INTERPRETER
    // --------------------------------------------------------

    gInterpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );


    if (gInterpreter == nullptr) {

        gStatus =
            "MNN interpreter creation failed";

        return JNI_FALSE;
    }


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

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


    if (preferOpenCl == JNI_TRUE) {

        gBackend =
            "OpenCL / FP16";

    } else {

        gBackend =
            "CPU";
    }


    gStatus =
        "Sana MNN model initialized";


    return JNI_TRUE;
}


// ============================================================
// JNI IS INITIALIZED
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
// JNI BACKEND
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
// JNI STATUS
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
// JNI RELEASE
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
// JNI TRANSFORMER TEST
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
// JNI VAE TEST
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject /* thiz */,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl) {

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


    std::string result =
        runVaeDiagnostic(
            vae,
            cache,
            preferOpenCl == JNI_TRUE
        );


    return env->NewStringUTF(
        result.c_str()
    );
}


// ============================================================
// JNI BOTH MODELS TEST
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


    std::string result =
        runCombinedDiagnostic(
            transformer,
            vae,
            cache,
            preferOpenCl == JNI_TRUE
        );


    return env->NewStringUTF(
        result.c_str()
    );
}

} // namespace
