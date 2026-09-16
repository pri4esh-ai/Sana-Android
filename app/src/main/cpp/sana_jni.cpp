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
// JNI STRING
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// SHAPE
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// DIMENSION TYPE
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// BACKEND
// ------------------------------------------------------------

std::string backendName(
        bool preferOpenCl) {

    if (preferOpenCl) {
        return "OpenCL / FP16";
    }

    return "CPU";
}


// ------------------------------------------------------------
// DESTROY INTERPRETER
// ------------------------------------------------------------

void destroyInterpreter(
        MNN::Interpreter*& interpreter) {

    if (interpreter != nullptr) {

        MNN::Interpreter::destroy(
            interpreter
        );

        interpreter = nullptr;
    }
}


// ------------------------------------------------------------
// GLOBAL RELEASE
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// SESSION CONFIG
// ------------------------------------------------------------

MNN::ScheduleConfig makeScheduleConfig(
        bool preferOpenCl,
        int cpuThreads) {

    MNN::ScheduleConfig config;

    if (preferOpenCl) {

        config.type =
            MNN_FORWARD_OPENCL;

        /*
         * GPU uses mode.
         *
         * MNN ScheduleConfig uses a union:
         * CPU -> numThread
         * GPU -> mode
         */

        config.mode =
            MNN_GPU_TUNING_HEAVY;

    } else {

        /*
         * CPU only needs:
         *
         * type
         * numThread
         */

        config.type =
            MNN_FORWARD_CPU;

        config.numThread =
            cpuThreads;
    }

    return config;
}


// ------------------------------------------------------------
// CREATE SESSION
// ------------------------------------------------------------

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

    /*
     * CPU fallback for unsupported OpenCL operations.
     */

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


// ------------------------------------------------------------
// TRANSFORMER DIAGNOSTIC
// ------------------------------------------------------------

std::string runTransformerDiagnostic(
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl) {

    std::ostringstream result;

    result
        << "SANA 0.6B / 512 TRANSFORMER TEST\n\n";

    result
        << "Transformer file:\n"
        << modelPath
        << "\n\n";


    // --------------------------------------------------------
    // FILE
    // --------------------------------------------------------

    std::ifstream file(
        modelPath,
        std::ios::binary |
        std::ios::ate
    );

    if (!file.good()) {

        result
            << "FAIL: Transformer file does not exist.";

        return result.str();
    }

    std::streamsize fileSize =
        file.tellg();

    file.close();

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


    timestepData[0] = 0.5f;


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


// ------------------------------------------------------------
// VAE DIAGNOSTIC
// ------------------------------------------------------------

std::string runVaeDiagnostic(
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl) {

    std::ostringstream result;

    result
        << "VAE Decoder\n"
        << "===\n\n";


    result
        << "Model file:\n"
        << modelPath
        << "\n\n";


    // --------------------------------------------------------
    // FILE
    // --------------------------------------------------------

    std::ifstream file(
        modelPath,
        std::ios::binary |
        std::ios::ate
    );

    if (!file.good()) {

        result
            << "FAIL: VAE model file does not exist.";

        return result.str();
    }


    std::streamsize fileSize =
        file.tellg();

    file.close();


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
            << "FAIL: Could not create VAE interpreter.";

        return result.str();
    }


    result
        << "Interpreter created.\n\n";


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
            << "FAIL: Could not create VAE session.";

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
        << "Input: latent\n\n";


    result
        << "Original shape: "
        << formatShape(input)
        << "\n";


    result
        << "Original elements: "
        << input->elementSize()
        << "\n";


    result
        << "Original tensor type code: "
        << input->getType().code
        << "\n";


    result
        << "Original tensor bytes: "
        << input->getType().bytes()
        << "\n";


    result
        << "Original dimension type: "
        << dimensionTypeName(
            input->getDimensionType()
        )
        << "\n";


    result
        << "Original device ID: "
        << input->deviceId()
        << "\n\n";


    // --------------------------------------------------------
    // EXPECTED SANA LATENT
    // --------------------------------------------------------

    std::vector<int> expectedShape = {
        1,
        32,
        16,
        16
    };


    /*
     * Do NOT resize if already correct.
     */

    std::vector<int> currentShape =
        input->shape();


    if (currentShape != expectedShape) {

        result
            << "Resizing VAE input to [1, 32, 16, 16]...\n";


        interpreter->resizeTensor(
            input,
            expectedShape
        );


        interpreter->resizeSession(
            session
        );
    }


    result
        << "Resized shape: "
        << formatShape(input)
        << "\n";


    result
        << "Resized elements: "
        << input->elementSize()
        << "\n";


    result
        << "Resized tensor type code: "
        << input->getType().code
        << "\n";


    result
        << "Resized tensor bytes: "
        << input->getType().bytes()
        << "\n";


    result
        << "Resized dimension type: "
        << dimensionTypeName(
            input->getDimensionType()
        )
        << "\n";


    result
        << "Resized device ID: "
        << input->deviceId()
        << "\n\n";


    // --------------------------------------------------------
    // HOST TENSOR
    // --------------------------------------------------------

    result
        << "Creating host tensor for VAE input...\n";


    /*
     * IMPORTANT:
     *
     * Use the exact dimension type of the
     * MNN device input.
     *
     * We do NOT force CAFFE here.
     */

    MNN::Tensor* hostTensor =
        new MNN::Tensor(
            input,
            input->getDimensionType()
        );


    if (hostTensor == nullptr) {

        result
            << "FAIL: Could not create host tensor.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Host tensor shape: "
        << formatShape(hostTensor)
        << "\n";


    result
        << "Host tensor elements: "
        << hostTensor->elementSize()
        << "\n";


    result
        << "Host tensor bytes: "
        << hostTensor->getType().bytes()
        << "\n";


    result
        << "Host dimension type: "
        << dimensionTypeName(
            hostTensor->getDimensionType()
        )
        << "\n";


    result
        << "Host device ID: "
        << hostTensor->deviceId()
        << "\n";


    // --------------------------------------------------------
    // HOST MEMORY
    // --------------------------------------------------------

    float* hostLatent =
        hostTensor->host<float>();


    if (hostLatent == nullptr) {

        result
            << "\nFAIL: Host tensor has no host memory.";

        delete hostTensor;

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    int elementCount =
        hostTensor->elementSize();


    result
        << "\nHost latent elements: "
        << elementCount
        << "\n\n";


    // --------------------------------------------------------
    // TEST LATENT
    // --------------------------------------------------------

    for (int i = 0;
         i < elementCount;
         ++i) {

        hostLatent[i] =
            -0.05f +
            static_cast<float>(i) *
            0.000390625f;
    }


    if (elementCount >= 4) {

        result
            << "Latent sample: "
            << hostLatent[0]
            << ", "
            << hostLatent[1]
            << ", "
            << hostLatent[2]
            << ", "
            << hostLatent[3]
            << "\n";
    }


    // --------------------------------------------------------
    // COPY TO OPENCL
    // --------------------------------------------------------

    result
        << "\nCopying host tensor to OpenCL device...\n";


    bool copied =
        input->copyFromHostTensor(
            hostTensor
        );


    delete hostTensor;


    if (!copied) {

        result
            << "\nFAIL: MNN copyFromHostTensor() failed.\n";


        result
            << "\nInput dimension type: "
            << dimensionTypeName(
                input->getDimensionType()
            )
            << "\n";


        result
            << "Input type code: "
            << input->getType().code
            << "\n";


        result
            << "Input bytes: "
            << input->getType().bytes()
            << "\n";


        result
            << "Input device ID: "
            << input->deviceId()
            << "\n";


        result
            << "\nHost-to-device transfer failed "
               "before VAE inference.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Host tensor copied successfully.\n\n";


    // --------------------------------------------------------
    // RUN VAE
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
        << "\nTime: "
        << elapsed
        << " ms\n\n";


    result
        << "Error code: "
        << errorCode
        << "\n\n";


    if (errorCode != 0) {

        result
            << "FAIL: VAE inference error.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


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
            << "FAIL: VAE output tensor not found.";

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
        << output->getType().code
        << "\n";


    result
        << "Output bytes: "
        << output->getType().bytes()
        << "\n";


    result
        << "Output dimension type: "
        << dimensionTypeName(
            output->getDimensionType()
        )
        << "\n";


    // --------------------------------------------------------
    // OUTPUT HOST COPY
    // --------------------------------------------------------

    MNN::Tensor* outputHost =
        new MNN::Tensor(
            output,
            output->getDimensionType()
        );


    bool outputCopied =
        output->copyToHostTensor(
            outputHost
        );


    if (outputCopied &&
        outputHost->host<float>() != nullptr &&
        outputHost->elementSize() >= 4) {

        float* data =
            outputHost->host<float>();


        result
            << "\nOutput sample: "
            << data[0]
            << ", "
            << data[1]
            << ", "
            << data[2]
            << ", "
            << data[3]
            << "\n";

    } else {

        result
            << "\nOutput host copy failed "
               "or output is not float32.\n";
    }


    delete outputHost;


    result
        << "\nPASS: VAE inference executed successfully.";


    interpreter->releaseSession(
        session
    );


    destroyInterpreter(
        interpreter
    );


    return result.str();
}


// ------------------------------------------------------------
// COMBINED TEST
// ------------------------------------------------------------

std::string runCombinedDiagnostic(
        const std::string& transformerPath,
        const std::string& vaePath,
        const std::string& cachePath,
        bool preferOpenCl) {

    std::ostringstream result;


    result
        << runTransformerDiagnostic(
            transformerPath,
            cachePath,
            preferOpenCl
        );


    result
        << "\n\n"
        << "========================================"
        << "\n\n";


    result
        << runVaeDiagnostic(
            vaePath,
            cachePath,
            preferOpenCl
        );


    return result.str();
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
        jobject assetManager,
        jstring modelAsset,
        jstring cachePath,
        jboolean preferOpenCl,
        jint cpuThreads) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


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
            assetManager
        );


    if (manager == nullptr) {

        gStatus =
            "Could not obtain AssetManager";

        return JNI_FALSE;
    }


    std::string assetPath =
        jstringToString(
            env,
            modelAsset
        );


    std::string cache =
        jstringToString(
            env,
            cachePath
        );


    // --------------------------------------------------------
    // OPEN ASSET
    // --------------------------------------------------------

    AAsset* asset =
        AAssetManager_open(
            manager,
            assetPath.c_str(),
            AASSET_MODE_STREAMING
        );


    if (asset == nullptr) {

        gStatus =
            "Could not open model asset: " +
            assetPath;

        return JNI_FALSE;
    }


    // --------------------------------------------------------
    // CACHE MODEL
    // --------------------------------------------------------

    std::string modelPath =
        cache +
        "/sana_model.mnn";


    FILE* output =
        std::fopen(
            modelPath.c_str(),
            "wb"
        );


    if (output == nullptr) {

        AAsset_close(asset);

        gStatus =
            "Could not create cached model";

        return JNI_FALSE;
    }


    char buffer[
        1024 * 1024
    ];


    int bytesRead = 0;


    while ((bytesRead =
            AAsset_read(
                asset,
                buffer,
                sizeof(buffer)
            )) > 0) {

        std::fwrite(
            buffer,
            1,
            bytesRead,
            output
        );
    }


    std::fclose(output);

    AAsset_close(asset);


    // --------------------------------------------------------
    // CREATE INTERPRETER
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


    // --------------------------------------------------------
    // CREATE SESSION
    // --------------------------------------------------------

    if (!createSession(
            gInterpreter,
            gSession,
            preferOpenCl == JNI_TRUE,
            threads)) {

        MNN::Interpreter::destroy(
            gInterpreter
        );

        gInterpreter = nullptr;

        gStatus =
            "MNN session creation failed";

        return JNI_FALSE;
    }


    gInitialized = true;


    gBackend =
        backendName(
            preferOpenCl == JNI_TRUE
        );


    gStatus =
        "Sana MNN engine initialized";


    LOGI(
        "Sana initialized: %s",
        gBackend.c_str()
    );


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
        jobject) {

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
        jobject) {

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
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    releaseGlobalLocked();


    LOGI(
        "Sana engine released"
    );
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
        jobject,
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
        jobject,
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
