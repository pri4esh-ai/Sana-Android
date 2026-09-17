#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
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

std::string gBackend = "CPU";
std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;
MNN::Session* gSession = nullptr;


/* ============================================================
 * JNI STRING
 * ============================================================ */

std::string jstringToString(JNIEnv* env, jstring s) {

    if (!s) {
        return "";
    }

    const char* c =
        env->GetStringUTFChars(
            s,
            nullptr
        );

    std::string result =
        c ? c : "";

    if (c) {
        env->ReleaseStringUTFChars(
            s,
            c
        );
    }

    return result;
}


/* ============================================================
 * FILE SIZE
 * ============================================================ */

long long getFileSize(
    const std::string& path
) {

    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file.good()) {
        return -1;
    }

    return static_cast<long long>(
        file.tellg()
    );
}


/* ============================================================
 * FORMAT SHAPE
 * ============================================================ */

std::string formatShape(
    const std::vector<int>& shape
) {

    std::ostringstream out;

    out << "[";

    for (size_t i = 0; i < shape.size(); ++i) {

        if (i != 0) {
            out << ", ";
        }

        out << shape[i];
    }

    out << "]";

    return out.str();
}


/* ============================================================
 * SHA-256
 * ============================================================ */

class Sha256 {

private:

    uint32_t state[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };

    uint64_t bitCount = 0;

    unsigned char buffer[64] = {};

    size_t bufferLength = 0;


    static uint32_t rotateRight(
        uint32_t value,
        uint32_t amount
    ) {

        return
            (value >> amount) |
            (value << (32 - amount));
    }


    static uint32_t choose(
        uint32_t e,
        uint32_t f,
        uint32_t g
    ) {

        return
            (e & f) ^
            (~e & g);
    }


    static uint32_t majority(
        uint32_t a,
        uint32_t b,
        uint32_t c
    ) {

        return
            (a & b) ^
            (a & c) ^
            (b & c);
    }


    static uint32_t bigSigma0(
        uint32_t x
    ) {

        return
            rotateRight(x, 2) ^
            rotateRight(x, 13) ^
            rotateRight(x, 22);
    }


    static uint32_t bigSigma1(
        uint32_t x
    ) {

        return
            rotateRight(x, 6) ^
            rotateRight(x, 11) ^
            rotateRight(x, 25);
    }


    static uint32_t smallSigma0(
        uint32_t x
    ) {

        return
            rotateRight(x, 7) ^
            rotateRight(x, 18) ^
            (x >> 3);
    }


    static uint32_t smallSigma1(
        uint32_t x
    ) {

        return
            rotateRight(x, 17) ^
            rotateRight(x, 19) ^
            (x >> 10);
    }


    static uint32_t loadBigEndian(
        const unsigned char* p
    ) {

        return
            (static_cast<uint32_t>(p[0]) << 24) |
            (static_cast<uint32_t>(p[1]) << 16) |
            (static_cast<uint32_t>(p[2]) << 8) |
            static_cast<uint32_t>(p[3]);
    }


    void transform(
        const unsigned char block[64]
    ) {

        static const uint32_t K[64] = {

            0x428a2f98,
            0x71374491,
            0xb5c0fbcf,
            0xe9b5dba5,
            0x3956c25b,
            0x59f111f1,
            0x923f82a4,
            0xab1c5ed5,
            0xd807aa98,
            0x12835b01,
            0x243185be,
            0x550c7dc3,
            0x72be5d74,
            0x80deb1fe,
            0x9bdc06a7,
            0xc19bf174,
            0xe49b69c1,
            0xefbe4786,
            0x0fc19dc6,
            0x240ca1cc,
            0x2de92c6f,
            0x4a7484aa,
            0x5cb0a9dc,
            0x76f988da,
            0x983e5152,
            0xa831c66d,
            0xb00327c8,
            0xbf597fc7,
            0xc6e00bf3,
            0xd5a79147,
            0x06ca6351,
            0x14292967,
            0x27b70a85,
            0x2e1b2138,
            0x4d2c6dfc,
            0x53380d13,
            0x650a7354,
            0x766a0abb,
            0x81c2c92e,
            0x92722c85,
            0xa2bfe8a1,
            0xa81a664b,
            0xc24b8b70,
            0xc76c51a3,
            0xd192e819,
            0xd6990624,
            0xf40e3585,
            0x106aa070,
            0x19a4c116,
            0x1e376c08,
            0x2748774c,
            0x34b0bcb5,
            0x391c0cb3,
            0x4ed8aa4a,
            0x5b9cca4f,
            0x682e6ff3,
            0x748f82ee,
            0x78a5636f,
            0x84c87814,
            0x8cc70208,
            0x90befffa,
            0xa4506ceb,
            0xbef9a3f7,
            0xc67178f2
        };


        uint32_t W[64];


        for (int i = 0; i < 16; ++i) {

            W[i] =
                loadBigEndian(
                    block + i * 4
                );
        }


        /*
         * IMPORTANT:
         *
         * Correct expression is:
         *
         * W[i - 16]
         *
         * The previous version had a missing ']'
         * which caused the C++ compilation failure.
         */

        for (int i = 16; i < 64; ++i) {

            W[i] =
                smallSigma1(W[i - 2]) +
                W[i - 7] +
                smallSigma0(W[i - 15]) +
                W[i - 16];
        }


        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];


        for (int i = 0; i < 64; ++i) {

            uint32_t t1 =
                h +
                bigSigma1(e) +
                choose(e, f, g) +
                K[i] +
                W[i];

            uint32_t t2 =
                bigSigma0(a) +
                majority(a, b, c);


            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }


        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }


public:

    void update(
        const unsigned char* data,
        size_t size
    ) {

        bitCount +=
            static_cast<uint64_t>(size) * 8ULL;


        size_t offset = 0;


        while (offset < size) {

            size_t copySize =
                std::min(
                    static_cast<size_t>(
                        64 - bufferLength
                    ),
                    size - offset
                );


            std::memcpy(
                buffer + bufferLength,
                data + offset,
                copySize
            );


            bufferLength += copySize;
            offset += copySize;


            if (bufferLength == 64) {

                transform(buffer);

                bufferLength = 0;
            }
        }
    }


    std::string final() {

        uint64_t originalBits =
            bitCount;


        buffer[bufferLength++] =
            0x80;


        if (bufferLength > 56) {

            while (bufferLength < 64) {

                buffer[bufferLength++] =
                    0;
            }


            transform(buffer);

            bufferLength = 0;
        }


        while (bufferLength < 56) {

            buffer[bufferLength++] =
                0;
        }


        for (int i = 7; i >= 0; --i) {

            buffer[bufferLength++] =
                static_cast<unsigned char>(
                    originalBits >> (i * 8)
                );
        }


        transform(buffer);


        std::ostringstream out;


        for (int i = 0; i < 8; ++i) {

            out
                << std::hex
                << std::setw(8)
                << std::setfill('0')
                << state[i];
        }


        return out.str();
    }
};


/* ============================================================
 * CALCULATE SHA-256
 * ============================================================ */

std::string calculateSha256(
    const std::string& path
) {

    std::ifstream file(
        path,
        std::ios::binary
    );


    if (!file) {

        return "ERROR";
    }


    Sha256 sha;


    unsigned char buffer[
        1024 * 1024
    ];


    while (file.good()) {

        file.read(
            reinterpret_cast<char*>(buffer),
            sizeof(buffer)
        );


        std::streamsize count =
            file.gcount();


        if (count > 0) {

            sha.update(
                buffer,
                static_cast<size_t>(count)
            );
        }
    }


    return sha.final();
}


/* ============================================================
 * CPU SESSION
 * ============================================================ */

MNN::Session* createCpuSession(
    MNN::Interpreter* interpreter,
    int threads = 4
) {

    if (!interpreter) {

        return nullptr;
    }


    MNN::ScheduleConfig config;

    config.type =
        MNN_FORWARD_CPU;

    config.numThread =
        threads;


    return interpreter->createSession(
        config
    );
}


/* ============================================================
 * RELEASE GLOBAL SESSION
 * ============================================================ */

void releaseGlobalSession() {

    if (gInterpreter && gSession) {

        gInterpreter->releaseSession(
            gSession
        );

        gSession = nullptr;
    }


    if (gInterpreter) {

        delete gInterpreter;

        gInterpreter = nullptr;
    }


    gInitialized = false;
}


/* ============================================================
 * TRANSFORMER CPU TEST
 * ============================================================ */

std::string runTransformerCpuTest(
    const std::string& path
) {

    std::ostringstream result;


    result
        << "SANA TRANSFORMER CPU TEST\n\n";


    result
        << "Model: "
        << path
        << "\n";


    if (getFileSize(path) <= 0) {

        result
            << "File missing.";

        return result.str();
    }


    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            path.c_str()
        );


    if (!interpreter) {

        result
            << "Interpreter FAIL";

        return result.str();
    }


    MNN::Session* session =
        createCpuSession(
            interpreter,
            4
        );


    if (!session) {

        delete interpreter;

        result
            << "Session FAIL";

        return result.str();
    }


    MNN::Tensor* hiddenStates =
        interpreter->getSessionInput(
            session,
            "hidden_states"
        );


    MNN::Tensor* timestep =
        interpreter->getSessionInput(
            session,
            "timestep"
        );


    MNN::Tensor* encoderHiddenStates =
        interpreter->getSessionInput(
            session,
            "encoder_hidden_states"
        );


    if (!hiddenStates ||
        !timestep ||
        !encoderHiddenStates) {

        interpreter->releaseSession(
            session
        );

        delete interpreter;


        result
            << "Inputs FAIL";

        return result.str();
    }


    MNN::Tensor hostHidden(
        hiddenStates,
        MNN::Tensor::CAFFE
    );


    MNN::Tensor hostTimestep(
        timestep,
        MNN::Tensor::CAFFE
    );


    MNN::Tensor hostEncoder(
        encoderHiddenStates,
        MNN::Tensor::CAFFE
    );


    if (!hostHidden.host<float>() ||
        !hostTimestep.host<float>() ||
        !hostEncoder.host<float>()) {

        interpreter->releaseSession(
            session
        );

        delete interpreter;


        result
            << "Host tensors FAIL";

        return result.str();
    }


    std::fill(
        hostHidden.host<float>(),
        hostHidden.host<float>() +
            hostHidden.elementSize(),
        0.0f
    );


    std::fill(
        hostTimestep.host<float>(),
        hostTimestep.host<float>() +
            hostTimestep.elementSize(),
        0.0f
    );


    std::fill(
        hostEncoder.host<float>(),
        hostEncoder.host<float>() +
            hostEncoder.elementSize(),
        0.0f
    );


    hiddenStates->copyFromHostTensor(
        &hostHidden
    );


    timestep->copyFromHostTensor(
        &hostTimestep
    );


    encoderHiddenStates->copyFromHostTensor(
        &hostEncoder
    );


    auto start =
        std::chrono::steady_clock::now();


    MNN::ErrorCode code =
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
        << " ms\n";


    result
        << "Error: "
        << static_cast<int>(code)
        << "\n";


    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            "sample"
        );


    if (output) {

        result
            << "Output: "
            << formatShape(
                output->shape()
            )
            << "\n";
    }


    interpreter->releaseSession(
        session
    );


    delete interpreter;


    return result.str();
}


/* ============================================================
 * VAE CPU ZERO LATENT TEST
 * ============================================================ */

std::string runVaeCpuZeroLatentTest(
    const std::string& path
) {

    std::ostringstream result;


    result
        << "SANA VAE FP32 CPU TEST\n\n";


    result
        << "Model:\n"
        << path
        << "\n\n";


    long long fileSize =
        getFileSize(path);


    result
        << "File size: "
        << fileSize
        << " bytes\n\n";


    if (fileSize <= 0) {

        result
            << "File missing.";

        return result.str();
    }


    result
        << "Calculating VAE SHA-256...\n";


    auto shaStart =
        std::chrono::steady_clock::now();


    std::string sha =
        calculateSha256(path);


    auto shaEnd =
        std::chrono::steady_clock::now();


    double shaTime =
        std::chrono::duration<double, std::milli>(
            shaEnd - shaStart
        ).count();


    result
        << "SHA-256 calculation time: "
        << shaTime
        << " ms\n\n";


    result
        << "VAE SHA256:\n"
        << sha
        << "\n\n";


    result
        << "Creating MNN interpreter...\n";


    auto interpreterStart =
        std::chrono::steady_clock::now();


    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            path.c_str()
        );


    auto interpreterEnd =
        std::chrono::steady_clock::now();


    double interpreterTime =
        std::chrono::duration<double, std::milli>(
            interpreterEnd - interpreterStart
        ).count();


    result
        << "Interpreter creation time: "
        << interpreterTime
        << " ms\n";


    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        return result.str();
    }


    result
        << "Interpreter: PASS\n\n";


    result
        << "Creating CPU session...\n";


    auto sessionStart =
        std::chrono::steady_clock::now();


    MNN::Session* session =
        createCpuSession(
            interpreter,
            4
        );


    auto sessionEnd =
        std::chrono::steady_clock::now();


    double sessionTime =
        std::chrono::duration<double, std::milli>(
            sessionEnd - sessionStart
        ).count();


    result
        << "Session creation time: "
        << sessionTime
        << " ms\n";


    if (!session) {

        result
            << "CPU session: FAIL\n";


        delete interpreter;


        return result.str();
    }


    result
        << "CPU session: PASS\n\n";


    MNN::Tensor* input =
        interpreter->getSessionInput(
            session,
            nullptr
        );


    if (!input) {

        result
            << "Input tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    std::vector<int> shape =
        input->shape();


    result
        << "Input shape: "
        << formatShape(shape)
        << "\n";


    if (shape.size() != 4 ||
        shape[0] != 1 ||
        shape[1] != 32 ||
        shape[2] != 16 ||
        shape[3] != 16) {

        result
            << "Input shape: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    result
        << "Input shape: PASS\n";


    result
        << "Input elements: "
        << input->elementSize()
        << "\n";


    result
        << "Input type bytes: "
        << sizeof(float)
        << "\n";


    result
        << "Input format: CAFFE / NCHW\n\n";


    result
        << "Creating host tensor...\n";


    MNN::Tensor hostInput(
        input,
        MNN::Tensor::CAFFE
    );


    if (!hostInput.host<float>()) {

        result
            << "Host tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    result
        << "Host tensor: PASS\n\n";


    const size_t elementCount =
        hostInput.elementSize();


    float* hostData =
        hostInput.host<float>();


    /*
     * EXACT ZERO LATENT
     */

    std::fill(
        hostData,
        hostData + elementCount,
        0.0f
    );


    result
        << "ZERO LATENT TEST\n";


    result
        << "All latent values set to exactly 0.0\n";


    result
        << "Latent sample: ";


    size_t sampleCount =
        std::min(
            static_cast<size_t>(3),
            elementCount
        );


    for (size_t i = 0;
         i < sampleCount;
         ++i) {

        if (i != 0) {
            result << ", ";
        }

        result
            << hostData[i];
    }


    result
        << "\n";


    bool allZero =
        true;


    for (size_t i = 0;
         i < elementCount;
         ++i) {

        if (hostData[i] != 0.0f) {

            allZero =
                false;

            break;
        }
    }


    result
        << "Latent zero verification: "
        << (allZero ? "PASS" : "FAIL")
        << "\n\n";


    if (!allZero) {

        result
            << "ZERO LATENT TEST: FAILED TO INITIALIZE\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    result
        << "Copying ZERO latent to MNN...\n";


    input->copyFromHostTensor(
        &hostInput
    );


    result
        << "Host -> MNN: PASS\n\n";


    result
        << "Running VAE inference...\n";


    auto inferenceStart =
        std::chrono::steady_clock::now();


    MNN::ErrorCode code =
        interpreter->runSession(
            session
        );


    auto inferenceEnd =
        std::chrono::steady_clock::now();


    double inferenceTime =
        std::chrono::duration<double, std::milli>(
            inferenceEnd - inferenceStart
        ).count();


    result
        << "Inference time: "
        << inferenceTime
        << " ms\n";


    result
        << "MNN error code: "
        << static_cast<int>(code)
        << "\n";


    if (code != MNN::NO_ERROR) {

        result
            << "Inference: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    result
        << "Inference: PASS\n\n";


    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            nullptr
        );


    if (!output) {

        result
            << "Output tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    std::vector<int> outputShape =
        output->shape();


    result
        << "Output shape: "
        << formatShape(outputShape)
        << "\n";


    if (outputShape.size() != 4 ||
        outputShape[0] != 1 ||
        outputShape[1] != 3 ||
        outputShape[2] != 512 ||
        outputShape[3] != 512) {

        result
            << "Output shape: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    result
        << "Output shape: PASS\n";


    size_t outputElements =
        output->elementSize();


    result
        << "Output elements: "
        << outputElements
        << "\n";


    MNN::Tensor hostOutput(
        output,
        MNN::Tensor::CAFFE
    );


    float* outputData =
        hostOutput.host<float>();


    if (!outputData) {

        result
            << "Output host tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    size_t nanCount = 0;
    size_t infCount = 0;


    float minValue =
        std::numeric_limits<float>::infinity();


    float maxValue =
        -std::numeric_limits<float>::infinity();


    for (size_t i = 0;
         i < outputElements;
         ++i) {

        float value =
            outputData[i];


        if (std::isnan(value)) {

            ++nanCount;

        } else if (std::isinf(value)) {

            ++infCount;

        } else {

            minValue =
                std::min(
                    minValue,
                    value
                );


            maxValue =
                std::max(
                    maxValue,
                    value
                );
        }
    }


    result
        << "Output[0]: "
        << outputData[0]
        << "\n";


    result
        << "NaN count: "
        << nanCount
        << "\n";


    result
        << "Inf count: "
        << infCount
        << "\n";


    if (nanCount == 0 &&
        infCount == 0) {

        result
            << "Output min: "
            << minValue
            << "\n";


        result
            << "Output max: "
            << maxValue
            << "\n";


        result
            << "Output finite: YES\n";


        result
            << "ZERO LATENT VAE TEST: PASS\n";

    } else {

        result
            << "Output min: nan\n";


        result
            << "Output max: nan\n";


        result
            << "Output finite: NO\n";


        result
            << "ZERO LATENT VAE TEST: FAILED NUMERIC OUTPUT\n";


        result
            << "Diagnostic mode\n";
    }


    interpreter->releaseSession(
        session
    );


    delete interpreter;


    return result.str();
}


/* ============================================================
 * VAE OPENCL ZERO LATENT TEST
 * ============================================================ */

std::string runVaeOpenClZeroLatentTest(
    const std::string& path
) {

    std::ostringstream result;


    result
        << "SANA VAE OPENCL ZERO LATENT TEST\n\n";


    result
        << "Model:\n"
        << path
        << "\n\n";


    if (getFileSize(path) <= 0) {

        result
            << "File missing.";

        return result.str();
    }


    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            path.c_str()
        );


    if (!interpreter) {

        result
            << "Interpreter: FAIL";

        return result.str();
    }


    MNN::ScheduleConfig config;


    config.type =
        MNN_FORWARD_OPENCL;


    config.numThread =
        4;


    MNN::BackendConfig backendConfig;


    backendConfig.precision =
        MNN::BackendConfig::Precision_Low;


    backendConfig.power =
        MNN::BackendConfig::Power_Normal;


    config.backendConfig =
        &backendConfig;


    result
        << "Creating OpenCL session...\n";


    MNN::Session* session =
        interpreter->createSession(
            config
        );


    if (!session) {

        result
            << "OpenCL session: FAIL\n";


        delete interpreter;


        return result.str();
    }


    result
        << "OpenCL session: PASS\n";


    MNN::Tensor* input =
        interpreter->getSessionInput(
            session,
            nullptr
        );


    if (!input) {

        result
            << "Input tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    result
        << "Input shape: "
        << formatShape(
            input->shape()
        )
        << "\n";


    MNN::Tensor hostInput(
        input,
        MNN::Tensor::CAFFE
    );


    if (!hostInput.host<float>()) {

        result
            << "Host tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    std::fill(
        hostInput.host<float>(),
        hostInput.host<float>() +
            hostInput.elementSize(),
        0.0f
    );


    input->copyFromHostTensor(
        &hostInput
    );


    result
        << "Zero latent upload: PASS\n";


    auto start =
        std::chrono::steady_clock::now();


    MNN::ErrorCode code =
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
        << "Inference time: "
        << elapsed
        << " ms\n";


    result
        << "MNN error code: "
        << static_cast<int>(code)
        << "\n";


    if (code != MNN::NO_ERROR) {

        result
            << "OpenCL inference: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            nullptr
        );


    if (!output) {

        result
            << "Output: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    MNN::Tensor hostOutput(
        output,
        MNN::Tensor::CAFFE
    );


    float* data =
        hostOutput.host<float>();


    if (!data) {

        result
            << "Output host tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );


        delete interpreter;


        return result.str();
    }


    size_t elements =
        hostOutput.elementSize();


    size_t nanCount = 0;
    size_t infCount = 0;


    float minValue =
        std::numeric_limits<float>::infinity();


    float maxValue =
        -std::numeric_limits<float>::infinity();


    for (size_t i = 0;
         i < elements;
         ++i) {

        float value =
            data[i];


        if (std::isnan(value)) {

            ++nanCount;

        } else if (std::isinf(value)) {

            ++infCount;

        } else {

            minValue =
                std::min(
                    minValue,
                    value
                );


            maxValue =
                std::max(
                    maxValue,
                    value
                );
        }
    }


    result
        << "Output shape: "
        << formatShape(
            output->shape()
        )
        << "\n";


    result
        << "Output[0]: "
        << data[0]
        << "\n";


    result
        << "NaN count: "
        << nanCount
        << "\n";


    result
        << "Inf count: "
        << infCount
        << "\n";


    if (nanCount == 0 &&
        infCount == 0) {

        result
            << "Output min: "
            << minValue
            << "\n";


        result
            << "Output max: "
            << maxValue
            << "\n";


        result
            << "OpenCL numeric output: PASS\n";

    } else {

        result
            << "OpenCL numeric output: FAIL\n";
    }


    interpreter->releaseSession(
        session
    );


    delete interpreter;


    return result.str();
}

} // namespace


/* ============================================================
 * nativeInitialize
 * ============================================================ */

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

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    (void)env;
    (void)assetManager;
    (void)modelAsset;
    (void)cachePath;
    (void)preferOpenCl;
    (void)cpuThreads;


    releaseGlobalSession();


    gBackend =
        "CPU";


    gStatus =
        "Native Sana engine initialized";


    gInitialized =
        true;


    return JNI_TRUE;
}


/* ============================================================
 * nativeIsInitialized
 * ============================================================ */

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
    JNIEnv*,
    jobject
) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    return gInitialized
        ? JNI_TRUE
        : JNI_FALSE;
}


/* ============================================================
 * nativeGetBackend
 * ============================================================ */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
    JNIEnv* env,
    jobject
) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    return env->NewStringUTF(
        gBackend.c_str()
    );
}


/* ============================================================
 * nativeGetStatus
 * ============================================================ */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
    JNIEnv* env,
    jobject
) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    return env->NewStringUTF(
        gStatus.c_str()
    );
}


/* ============================================================
 * nativeRelease
 * ============================================================ */

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
    JNIEnv*,
    jobject
) {

    std::lock_guard<std::mutex> lock(
        gMutex
    );


    releaseGlobalSession();


    gBackend =
        "CPU";


    gStatus =
        "Released";
}


/* ============================================================
 * nativeTestTransformer
 * ============================================================ */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
    JNIEnv* env,
    jobject,
    jstring transformerPath,
    jstring cachePath,
    jboolean preferOpenCl
) {

    (void)cachePath;
    (void)preferOpenCl;


    std::string path =
        jstringToString(
            env,
            transformerPath
        );


    std::string result =
        runTransformerCpuTest(
            path
        );


    return env->NewStringUTF(
        result.c_str()
    );
}


/* ============================================================
 * nativeTestVae
 * ============================================================ */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
    JNIEnv* env,
    jobject,
    jstring vaePath,
    jstring cachePath,
    jboolean preferOpenCl
) {

    (void)cachePath;
    (void)preferOpenCl;


    std::string path =
        jstringToString(
            env,
            vaePath
        );


    /*
     * CPU diagnostic intentionally kept here.
     *
     * We are checking whether the VAE itself
     * produces finite values with an exact
     * zero latent.
     */

    std::string result =
        runVaeCpuZeroLatentTest(
            path
        );


    return env->NewStringUTF(
        result.c_str()
    );
}


/* ============================================================
 * nativeTestModels
 * ============================================================ */

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

    (void)cachePath;
    (void)preferOpenCl;


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


    std::ostringstream result;


    result
        << "SANA MODEL DIAGNOSTIC\n\n";


    result
        << "TRANSFORMER\n";


    result
        << runTransformerCpuTest(
            transformer
        );


    result
        << "\nVAE\n";


    result
        << runVaeCpuZeroLatentTest(
            vae
        );


    return env->NewStringUTF(
        result.str().c_str()
    );
}
