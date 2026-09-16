#include <jni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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
// BACKEND
// ============================================================

std::string backendName(
        bool preferOpenCl) {

    if (preferOpenCl) {
        return "OpenCL / FP16";
    }

    return "CPU";
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
// READ FILE SIZE
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


// ============================================================
// VAE CPU DIRECT-HOST TEST
// ============================================================

std::string runVaeCpuIsolationTest(
        const std::string& modelPath) {

    std::ostringstream result;

    result
        << "\n\n"
        << "========================================\n"
        << "VAE CPU DIRECT-HOST TEST\n"
        << "========================================\n\n";

    result
        << "This test bypasses OpenCL completely.\n"
        << "CPU input is filled directly through host memory.\n\n";


    // --------------------------------------------------------
    // CREATE CPU INTERPRETER
    // --------------------------------------------------------

    result
        << "Creating CPU interpreter...\n";

    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );

    if (interpreter == nullptr) {

        result
            << "FAIL: CPU VAE interpreter creation failed.";

        return result.str();
    }

    result
        << "CPU interpreter created.\n";


    // --------------------------------------------------------
    // CPU SESSION
    // --------------------------------------------------------

    MNN::Session* session = nullptr;

    if (!createSession(
            interpreter,
            session,
            false,
            4)) {

        result
            << "FAIL: CPU VAE session creation failed.";

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }

    result
        << "CPU session created.\n\n";


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
            << "FAIL: CPU VAE latent input not found.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "CPU input shape: "
        << formatShape(input)
        << "\n";

    result
        << "CPU input elements: "
        << input->elementSize()
        << "\n";

    result
        << "CPU input type code: "
        << static_cast<int>(
            input->getType().code
        )
        << "\n";

    result
        << "CPU input bytes: "
        << static_cast<int>(
            input->getType().bytes()
        )
        << "\n";

    result
        << "CPU input dimension type: "
        << dimensionTypeName(
            input->getDimensionType()
        )
        << "\n\n";


    // --------------------------------------------------------
    // EXPECTED SHAPE
    // --------------------------------------------------------

    std::vector<int> expectedShape = {
        1, 32, 16, 16
    };


    if (input->shape() != expectedShape) {

        result
            << "Applying CPU VAE resize...\n";

        interpreter->resizeTensor(
            input,
            expectedShape
        );

        interpreter->resizeSession(
            session
        );
    }


    result
        << "Final CPU input shape: "
        << formatShape(input)
        << "\n\n";


    // --------------------------------------------------------
    // DIRECT HOST POINTER
    // --------------------------------------------------------

    result
        << "Getting CPU host pointer...\n";

    float* host =
        input->host<float>();


    if (host == nullptr) {

        result
            << "FAIL: CPU input host pointer is null.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "CPU host pointer acquired.\n";


    // --------------------------------------------------------
    // FILL LATENT
    // --------------------------------------------------------

    const int elements =
        input->elementSize();

    for (int i = 0;
         i < elements;
         ++i) {

        host[i] =
            -0.05f +
            0.000390625f *
            static_cast<float>(
                i % 256
            );
    }


    result
        << "CPU latent sample: "
        << host[0]
        << ", "
        << host[1]
        << ", "
        << host[2]
        << ", "
        << host[3]
        << "\n\n";


    // --------------------------------------------------------
    // RUN CPU
    // --------------------------------------------------------

    result
        << "Running CPU VAE inference...\n";

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
        << "CPU VAE time: "
        << elapsed
        << " ms\n";

    result
        << "CPU VAE error code: "
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
            << "FAIL: CPU VAE output is null.";

    } else {

        result
            << "CPU VAE output shape: "
            << formatShape(output)
            << "\n";

        result
            << "CPU VAE output elements: "
            << output->elementSize()
            << "\n\n";


        if (errorCode == 0) {

            float* outputHost =
                output->host<float>();

            if (outputHost != nullptr) {

                result
                    << "CPU output sample: "
                    << outputHost[0]
                    << ", "
                    << outputHost[1]
                    << ", "
                    << outputHost[2]
                    << ", "
                    << outputHost[3]
                    << "\n\n";
            }

            result
                << "PASS: VAE executes on CPU.\n";

        } else {

            result
                << "FAIL: VAE CPU inference error.\n";
        }
    }


    interpreter->releaseSession(
        session
    );

    destroyInterpreter(
        interpreter
    );


    result
        << "\nCPU VAE test finished.";

    return result.str();
}


// ============================================================
// VAE OPENCL MAP TEST
// ============================================================

std::string runVaeOpenClMapTest(
        const std::string& modelPath) {

    std::ostringstream result;

    result
        << "\n\n"
        << "========================================\n"
        << "VAE OPENCL MAP TEST\n"
        << "========================================\n\n";

    result
        << "Testing MNN OpenCL input map/unmap path.\n";

    result
        << "This avoids copyFromHostTensor().\n\n";


    // --------------------------------------------------------
    // INTERPRETER
    // --------------------------------------------------------

    result
        << "Creating OpenCL interpreter...\n";

    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );

    if (interpreter == nullptr) {

        result
            << "FAIL: OpenCL VAE interpreter creation failed.";

        return result.str();
    }

    result
        << "OpenCL interpreter created.\n";


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

    MNN::Session* session = nullptr;

    if (!createSession(
            interpreter,
            session,
            true,
            4)) {

        result
            << "FAIL: OpenCL VAE session creation failed.";

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }

    result
        << "OpenCL session created.\n\n";


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
            << "FAIL: OpenCL VAE latent input not found.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "Original shape: "
        << formatShape(input)
        << "\n";

    result
        << "Original elements: "
        << input->elementSize()
        << "\n";

    result
        << "Original type code: "
        << static_cast<int>(
            input->getType().code
        )
        << "\n";

    result
        << "Original bytes: "
        << static_cast<int>(
            input->getType().bytes()
        )
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
    // RESIZE
    // --------------------------------------------------------

    std::vector<int> expectedShape = {
        1, 32, 16, 16
    };


    result
        << "Applying explicit VAE tensor resize...\n";

    interpreter->resizeTensor(
        input,
        expectedShape
    );

    result
        << "Calling resizeSession()...\n";

    interpreter->resizeSession(
        session
    );

    result
        << "resizeSession() completed.\n\n";


    result
        << "Resized shape: "
        << formatShape(input)
        << "\n";

    result
        << "Resized elements: "
        << input->elementSize()
        << "\n";

    result
        << "Resized type code: "
        << static_cast<int>(
            input->getType().code
        )
        << "\n";

    result
        << "Resized bytes: "
        << static_cast<int>(
            input->getType().bytes()
        )
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
    // WAIT
    // --------------------------------------------------------

    result
        << "Testing OpenCL input wait/write state...\n";

    int waitResult =
        input->wait(
            MNN::Tensor::MAP_TENSOR_WRITE,
            true
        );

    result
        << "Input wait(MAP_TENSOR_WRITE, true): "
        << waitResult
        << "\n\n";


    // --------------------------------------------------------
    // MAP
    // --------------------------------------------------------

    result
        << "Mapping OpenCL input for WRITE...\n";

    MNN::Tensor::DimensionType dimensionType =
        input->getDimensionType();

    void* mapped =
        input->map(
            MNN::Tensor::MAP_TENSOR_WRITE,
            dimensionType
        );


    if (mapped == nullptr) {

        result
            << "FAIL: MNN could not map VAE input tensor for WRITE.\n\n";

        result
            << "This means the OpenCL backend did not expose a writable "
            << "mapped input buffer for this session.\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    result
        << "PASS: OpenCL input mapped successfully.\n";


    // --------------------------------------------------------
    // FILL MAPPED BUFFER
    // --------------------------------------------------------

    float* data =
        static_cast<float*>(mapped);


    if (data == nullptr) {

        input->unmap(
            MNN::Tensor::MAP_TENSOR_WRITE,
            dimensionType,
            mapped
        );

        result
            << "FAIL: Mapped pointer is null.";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


    const int elements =
        input->elementSize();


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


    result
        << "Mapped latent sample: "
        << data[0]
        << ", "
        << data[1]
        << ", "
        << data[2]
        << ", "
        << data[3]
        << "\n";


    // --------------------------------------------------------
    // UNMAP
    // --------------------------------------------------------

    result
        << "Unmapping OpenCL input...\n";

    input->unmap(
        MNN::Tensor::MAP_TENSOR_WRITE,
        dimensionType,
        mapped
    );

    result
        << "OpenCL input unmap completed.\n\n";


    // --------------------------------------------------------
    // RUN
    // --------------------------------------------------------

    result
        << "Running OpenCL VAE inference...\n";

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
        << "OpenCL VAE time: "
        << elapsed
        << " ms\n";

    result
        << "OpenCL VAE error code: "
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
            << "FAIL: OpenCL VAE output is null.";

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


    // --------------------------------------------------------
    // OUTPUT MAP
    // --------------------------------------------------------

    result
        << "\nMapping OpenCL output for READ...\n";


    int outputWait =
        output->wait(
            MNN::Tensor::MAP_TENSOR_READ,
            true
        );


    result
        << "Output wait result: "
        << outputWait
        << "\n";


    void* outputMapped =
        output->map(
            MNN::Tensor::MAP_TENSOR_READ,
            output->getDimensionType()
        );


    if (outputMapped != nullptr) {

        float* outputData =
            static_cast<float*>(
                outputMapped
            );

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


        output->unmap(
            MNN::Tensor::MAP_TENSOR_READ,
            output->getDimensionType(),
            outputMapped
        );

        result
            << "Output unmap completed.\n";

    } else {

        result
            << "WARNING: Could not map OpenCL output.\n";

        /*
         * Try the official device -> host method.
         */
        MNN::Tensor* hostOutput =
            MNN::Tensor::createHostTensorFromDevice(
                output,
                true
            );

        if (hostOutput != nullptr) {

            float* outputData =
                hostOutput->host<float>();

            if (outputData != nullptr) {

                result
                    << "Host output sample: "
                    << outputData[0]
                    << ", "
                    << outputData[1]
                    << ", "
                    << outputData[2]
                    << ", "
                    << outputData[3]
                    << "\n";
            }

            MNN::Tensor::destroy(
                hostOutput
            );
        }
    }


    result
        << "\n";


    if (errorCode == 0) {

        result
            << "PASS: VAE executed successfully on OpenCL.\n";

    } else {

        result
            << "FAIL: OpenCL VAE inference error.\n";
    }


    interpreter->releaseSession(
        session
    );

    destroyInterpreter(
        interpreter
    );


    result
        << "\nOpenCL VAE test finished.";

    return result.str();
}


// ============================================================
// COMPLETE VAE DIAGNOSTIC
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


    /*
     * Always perform the CPU direct-host isolation test.
     *
     * This tells us whether the exported VAE itself can execute
     * without OpenCL.
     */
    result
        << runVaeCpuIsolationTest(
            modelPath
        );


    /*
     * Only perform the OpenCL map test when OpenCL was requested.
     */
    if (preferOpenCl) {

        result
            << "\n\n"
            << runVaeOpenClMapTest(
                modelPath
            );
    }


    return result.str();
}


// ============================================================
// COMBINED MODEL DIAGNOSTIC
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


    // --------------------------------------------------------
    // CREATE SESSION
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
