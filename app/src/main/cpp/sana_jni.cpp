#include <jni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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
std::string gBackend = "NONE";
std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;
MNN::Session* gSession = nullptr;


// ============================================================
// Basic helpers
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
        const std::vector<int>& shape) {

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


long long getFileSize(
        const std::string& path) {

    std::ifstream file(
            path,
            std::ios::binary | std::ios::ate);

    if (!file.good()) {
        return -1;
    }

    return static_cast<long long>(
            file.tellg());
}


bool isFiniteFloat(float value) {
    return std::isfinite(value);
}


// ============================================================
// Dependency-free SHA-256
//
// This is intentionally implemented here so that we do NOT
// need OpenSSL or any additional Android library.
// ============================================================

class Sha256 {

private:

    uint32_t state[8];

    uint64_t bitLength;

    unsigned char buffer[64];

    size_t bufferLength;


    static uint32_t rotr(
            uint32_t x,
            uint32_t n) {

        return (x >> n) |
               (x << (32 - n));
    }


    static uint32_t choose(
            uint32_t e,
            uint32_t f,
            uint32_t g) {

        return (e & f) ^
               (~e & g);
    }


    static uint32_t majority(
            uint32_t a,
            uint32_t b,
            uint32_t c) {

        return (a & b) ^
               (a & c) ^
               (b & c);
    }


    static uint32_t bigSigma0(
            uint32_t x) {

        return rotr(x, 2) ^
               rotr(x, 13) ^
               rotr(x, 22);
    }


    static uint32_t bigSigma1(
            uint32_t x) {

        return rotr(x, 6) ^
               rotr(x, 11) ^
               rotr(x, 25);
    }


    static uint32_t smallSigma0(
            uint32_t x) {

        return rotr(x, 7) ^
               rotr(x, 18) ^
               (x >> 3);
    }


    static uint32_t smallSigma1(
            uint32_t x) {

        return rotr(x, 17) ^
               rotr(x, 19) ^
               (x >> 10);
    }


    static uint32_t loadBigEndian(
            const unsigned char* p) {

        return
                (static_cast<uint32_t>(p[0]) << 24) |
                (static_cast<uint32_t>(p[1]) << 16) |
                (static_cast<uint32_t>(p[2]) << 8) |
                (static_cast<uint32_t>(p[3]));
    }


    static void storeBigEndian(
            uint32_t value,
            unsigned char* p) {

        p[0] =
                static_cast<unsigned char>(
                        value >> 24);

        p[1] =
                static_cast<unsigned char>(
                        value >> 16);

        p[2] =
                static_cast<unsigned char>(
                        value >> 8);

        p[3] =
                static_cast<unsigned char>(
                        value);
    }


    void transform(
            const unsigned char block[64]) {

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
                            block + (i * 4));
        }


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

            uint32_t T1 =
                    h +
                    bigSigma1(e) +
                    choose(e, f, g) +
                    K[i] +
                    W[i];

            uint32_t T2 =
                    bigSigma0(a) +
                    majority(a, b, c);

            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
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

    Sha256()
            : bitLength(0),
              bufferLength(0) {

        state[0] = 0x6a09e667;
        state[1] = 0xbb67ae85;
        state[2] = 0x3c6ef372;
        state[3] = 0xa54ff53a;
        state[4] = 0x510e527f;
        state[5] = 0x9b05688c;
        state[6] = 0x1f83d9ab;
        state[7] = 0x5be0cd19;

        std::memset(
                buffer,
                0,
                sizeof(buffer));
    }


    void update(
            const unsigned char* data,
            size_t length) {

        if (data == nullptr ||
            length == 0) {

            return;
        }


        bitLength +=
                static_cast<uint64_t>(length) *
                8ULL;


        size_t offset = 0;


        while (offset < length) {

            size_t available =
                    64 - bufferLength;

            size_t copySize =
                    std::min(
                            available,
                            length - offset);


            std::memcpy(
                    buffer + bufferLength,
                    data + offset,
                    copySize);


            bufferLength += copySize;
            offset += copySize;


            if (bufferLength == 64) {

                transform(buffer);

                bufferLength = 0;
            }
        }
    }


    std::string final() {

        uint64_t originalBitLength =
                bitLength;


        buffer[bufferLength++] = 0x80;


        if (bufferLength > 56) {

            while (bufferLength < 64) {

                buffer[bufferLength++] = 0;
            }

            transform(buffer);

            bufferLength = 0;
        }


        while (bufferLength < 56) {

            buffer[bufferLength++] = 0;
        }


        for (int i = 7; i >= 0; --i) {

            buffer[bufferLength++] =
                    static_cast<unsigned char>(
                            originalBitLength >>
                            (i * 8));
        }


        transform(buffer);


        unsigned char digest[32];


        for (int i = 0; i < 8; ++i) {

            storeBigEndian(
                    state[i],
                    digest + (i * 4));
        }


        std::ostringstream output;


        for (int i = 0; i < 32; ++i) {

            output
                    << std::hex
                    << std::setw(2)
                    << std::setfill('0')
                    << static_cast<int>(
                            digest[i]);
        }


        return output.str();
    }
};


std::string calculateSha256(
        const std::string& filePath) {

    std::ifstream file(
            filePath,
            std::ios::binary);

    if (!file.is_open()) {

        return "ERROR: unable to open file";
    }


    Sha256 sha;


    unsigned char buffer[1024 * 1024];


    while (file.good()) {

        file.read(
                reinterpret_cast<char*>(buffer),
                sizeof(buffer));


        std::streamsize bytes =
                file.gcount();


        if (bytes > 0) {

            sha.update(
                    buffer,
                    static_cast<size_t>(
                            bytes));
        }
    }


    if (file.bad()) {

        return "ERROR: file read failed";
    }


    return sha.final();
}


// ============================================================
// CPU schedule
// ============================================================

MNN::ScheduleConfig makeCpuScheduleConfig(
        int cpuThreads = 4) {

    MNN::ScheduleConfig config;

    config.type =
            MNN_FORWARD_CPU;

    config.numThread =
            cpuThreads;

    return config;
}


// ============================================================
// OpenCL schedule
// ============================================================

MNN::ScheduleConfig makeOpenClScheduleConfig() {

    MNN::ScheduleConfig config;

    config.type =
            MNN_FORWARD_OPENCL;

    config.numThread =
            4;

    config.mode =
            MNN_GPU_TUNING_HEAVY;

    config.backupType =
            MNN_FORWARD_CPU;

    return config;
}


// ============================================================
// Cleanup
// ============================================================

void releaseGlobalSession() {

    if (gInterpreter != nullptr &&
        gSession != nullptr) {

        gInterpreter->releaseSession(
                gSession);

        gSession = nullptr;
    }


    if (gInterpreter != nullptr) {

        delete gInterpreter;

        gInterpreter = nullptr;
    }


    gInitialized = false;

    gBackend = "NONE";
}


// ============================================================
// CPU session
// ============================================================

MNN::Session* createCpuSession(
        MNN::Interpreter* interpreter,
        int cpuThreads = 4) {

    if (interpreter == nullptr) {
        return nullptr;
    }


    MNN::ScheduleConfig config =
            makeCpuScheduleConfig(
                    cpuThreads);


    return interpreter->createSession(
            config);
}


// ============================================================
// OpenCL session
// ============================================================

MNN::Session* createOpenClSession(
        MNN::Interpreter* interpreter) {

    if (interpreter == nullptr) {
        return nullptr;
    }


    MNN::ScheduleConfig config =
            makeOpenClScheduleConfig();


    return interpreter->createSession(
            config);
}


// ============================================================
// Transformer CPU TEST
// ============================================================

std::string runTransformerCpuTest(
        const std::string& transformerPath) {

    std::ostringstream report;


    report
            << "============================================\n"
            << "SANA TRANSFORMER CPU TEST\n"
            << "============================================\n";


    report
            << "Model:\n"
            << transformerPath
            << "\n";


    long long fileSize =
            getFileSize(
                    transformerPath);


    if (fileSize <= 0) {

        report
                << "File: FAIL\n"
                << "File does not exist or is empty.\n";

        return report.str();
    }


    report
            << "File size: "
            << fileSize
            << " bytes\n\n";


    auto createStart =
            std::chrono::steady_clock::now();


    report
            << "Creating MNN interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    transformerPath.c_str());


    auto createEnd =
            std::chrono::steady_clock::now();


    double createMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    createEnd -
                    createStart)
                    .count();


    report
            << "Interpreter creation time: "
            << createMs
            << " ms\n";


    if (interpreter == nullptr) {

        report
                << "Interpreter: FAIL\n";

        return report.str();
    }


    report
            << "Interpreter: PASS\n";


    auto sessionStart =
            std::chrono::steady_clock::now();


    MNN::Session* session =
            createCpuSession(
                    interpreter,
                    4);


    auto sessionEnd =
            std::chrono::steady_clock::now();


    double sessionMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    sessionEnd -
                    sessionStart)
                    .count();


    report
            << "Creating CPU session...\n"
            << "Session creation time: "
            << sessionMs
            << " ms\n";


    if (session == nullptr) {

        report
                << "CPU session: FAIL\n";

        delete interpreter;

        return report.str();
    }


    report
            << "CPU session: PASS\n";


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

        report
                << "Required transformer inputs: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Transformer inputs: PASS\n";


    // --------------------------------------------------------
    // hidden_states
    // --------------------------------------------------------

    report
            << "hidden_states shape: "
            << formatShape(
                    hiddenStates->shape())
            << "\n";


    MNN::Tensor hiddenHost(
            hiddenStates,
            MNN::Tensor::CAFFE);


    float* hiddenPtr =
            hiddenHost.host<float>();


    if (hiddenPtr == nullptr) {

        report
                << "hidden_states host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::fill(
            hiddenPtr,
            hiddenPtr +
                    hiddenHost.elementSize(),
            0.0f);


    hiddenStates->copyFromHostTensor(
            &hiddenHost);


    // --------------------------------------------------------
    // timestep
    // --------------------------------------------------------

    MNN::Tensor timestepHost(
            timestep,
            MNN::Tensor::CAFFE);


    float* timestepPtr =
            timestepHost.host<float>();


    if (timestepPtr == nullptr) {

        report
                << "timestep host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::fill(
            timestepPtr,
            timestepPtr +
                    timestepHost.elementSize(),
            0.0f);


    timestep->copyFromHostTensor(
            &timestepHost);


    // --------------------------------------------------------
    // encoder_hidden_states
    // --------------------------------------------------------

    MNN::Tensor encoderHost(
            encoderHiddenStates,
            MNN::Tensor::CAFFE);


    float* encoderPtr =
            encoderHost.host<float>();


    if (encoderPtr == nullptr) {

        report
                << "encoder_hidden_states host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::fill(
            encoderPtr,
            encoderPtr +
                    encoderHost.elementSize(),
            0.0f);


    encoderHiddenStates->copyFromHostTensor(
            &encoderHost);


    report
            << "Input tensors: PASS\n";


    // --------------------------------------------------------
    // Run transformer
    // --------------------------------------------------------

    report
            << "Running transformer inference...\n";


    auto inferenceStart =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto inferenceEnd =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    inferenceEnd -
                    inferenceStart)
                    .count();


    report
            << "Inference time: "
            << inferenceMs
            << " ms\n"
            << "MNN error code: "
            << static_cast<int>(
                    result)
            << "\n";


    if (result != MNN::NO_ERROR) {

        report
                << "Inference: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Inference: PASS\n";


    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    "sample");


    if (output == nullptr) {

        report
                << "Output tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Output shape: "
            << formatShape(
                    output->shape())
            << "\n";


    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);


    output->copyToHostTensor(
            &outputHost);


    float* outputPtr =
            outputHost.host<float>();


    if (outputPtr != nullptr &&
        outputHost.elementSize() > 0) {

        report
                << "Output[0]: "
                << outputPtr[0]
                << "\n";


        bool finite = true;


        for (size_t i = 0;
             i < outputHost.elementSize();
             ++i) {

            if (!isFiniteFloat(
                    outputPtr[i])) {

                finite = false;

                break;
            }
        }


        report
                << "Output finite: "
                << (finite
                    ? "YES"
                    : "NO")
                << "\n";
    }


    interpreter->releaseSession(
            session);

    delete interpreter;


    report
            << "============================================\n"
            << "SANA TRANSFORMER CPU TEST: PASS\n"
            << "============================================\n";


    return report.str();
}


// ============================================================
// VAE CPU TEST
//
// ZERO LATENT ONLY.
//
// Transformer is NOT used.
// ============================================================

std::string runVaeCpuTest(
        const std::string& vaePath) {

    std::ostringstream report;


    report
            << "============================================\n"
            << "SANA VAE FP16 CPU TEST\n"
            << "============================================\n";


    report
            << "Model:\n"
            << vaePath
            << "\n";


    long long fileSize =
            getFileSize(
                    vaePath);


    if (fileSize <= 0) {

        report
                << "File: FAIL\n"
                << "File does not exist or is empty.\n";

        return report.str();
    }


    report
            << "File size: "
            << fileSize
            << " bytes\n";


    // ========================================================
    // SHA-256
    // ========================================================

    report
            << "Calculating VAE SHA-256...\n";


    auto shaStart =
            std::chrono::steady_clock::now();


    std::string sha256 =
            calculateSha256(
                    vaePath);


    auto shaEnd =
            std::chrono::steady_clock::now();


    double shaMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    shaEnd -
                    shaStart)
                    .count();


    report
            << "SHA-256 calculation time: "
            << shaMs
            << " ms\n";


    report
            << "VAE SHA256: "
            << sha256
            << "\n";


    // --------------------------------------------------------
    // Interpreter
    // --------------------------------------------------------

    report
            << "Creating MNN interpreter...\n";


    auto createStart =
            std::chrono::steady_clock::now();


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    auto createEnd =
            std::chrono::steady_clock::now();


    double createMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    createEnd -
                    createStart)
                    .count();


    report
            << "Interpreter creation time: "
            << createMs
            << " ms\n";


    if (interpreter == nullptr) {

        report
                << "Interpreter: FAIL\n";

        return report.str();
    }


    report
            << "Interpreter: PASS\n";


    // --------------------------------------------------------
    // CPU session
    // --------------------------------------------------------

    report
            << "Creating CPU session...\n";


    auto sessionStart =
            std::chrono::steady_clock::now();


    MNN::Session* session =
            createCpuSession(
                    interpreter,
                    4);


    auto sessionEnd =
            std::chrono::steady_clock::now();


    double sessionMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    sessionEnd -
                    sessionStart)
                    .count();


    report
            << "Session creation time: "
            << sessionMs
            << " ms\n";


    if (session == nullptr) {

        report
                << "CPU session: FAIL\n";

        delete interpreter;

        return report.str();
    }


    report
            << "CPU session: PASS\n";


    // --------------------------------------------------------
    // Get latent
    // --------------------------------------------------------

    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        report
                << "Input tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::vector<int> inputShape =
            input->shape();


    report
            << "Input shape: "
            << formatShape(
                    inputShape)
            << "\n";


    size_t inputElements =
            input->elementSize();


    report
            << "Input elements: "
            << inputElements
            << "\n";


    report
            << "Input type bytes: "
            << input->getType().bytes()
            << "\n";


    report
            << "Input format: CAFFE / NCHW\n";


    // --------------------------------------------------------
    // Validate latent shape
    // --------------------------------------------------------

    bool correctShape =
            inputShape.size() == 4 &&
            inputShape[0] == 1 &&
            inputShape[1] == 32 &&
            inputShape[2] == 16 &&
            inputShape[3] == 16;


    if (!correctShape) {

        report
                << "Input shape: FAIL\n"
                << "Expected [1, 32, 16, 16]\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Input shape: PASS\n";


    // --------------------------------------------------------
    // Host tensor
    // --------------------------------------------------------

    report
            << "Creating host tensor...\n";


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    float* hostPtr =
            hostTensor.host<float>();


    if (hostPtr == nullptr) {

        report
                << "Host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Host tensor: PASS\n";


    // ========================================================
    // ZERO LATENT
    // ========================================================

    std::fill(
            hostPtr,
            hostPtr +
                    hostTensor.elementSize(),
            0.0f);


    report
            << "ZERO LATENT TEST\n"
            << "All latent values set to exactly 0.0\n";


    report
            << "Latent sample: "
            << hostPtr[0]
            << ", "
            << hostPtr[1]
            << ", "
            << hostPtr[2]
            << "\n";


    // --------------------------------------------------------
    // Verify all zero
    // --------------------------------------------------------

    bool allZero = true;


    for (size_t i = 0;
         i < hostTensor.elementSize();
         ++i) {

        if (hostPtr[i] != 0.0f) {

            allZero = false;

            break;
        }
    }


    report
            << "Latent zero verification: "
            << (allZero
                ? "PASS"
                : "FAIL")
            << "\n";


    if (!allZero) {

        report
                << "Zero latent preparation failed.\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    // --------------------------------------------------------
    // Copy to MNN
    // --------------------------------------------------------

    report
            << "Copying ZERO latent to MNN...\n";


    input->copyFromHostTensor(
            &hostTensor);


    report
            << "Host -> MNN: PASS\n";


    // --------------------------------------------------------
    // Run VAE
    // --------------------------------------------------------

    report
            << "Running VAE inference...\n";


    auto inferenceStart =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto inferenceEnd =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    inferenceEnd -
                    inferenceStart)
                    .count();


    report
            << "Inference time: "
            << inferenceMs
            << " ms\n";


    report
            << "MNN error code: "
            << static_cast<int>(
                    result)
            << "\n";


    if (result != MNN::NO_ERROR) {

        report
                << "Inference: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Inference: PASS\n";


    // --------------------------------------------------------
    // Get output
    // --------------------------------------------------------

    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report
                << "Output tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::vector<int> outputShape =
            output->shape();


    report
            << "Output shape: "
            << formatShape(
                    outputShape)
            << "\n";


    size_t outputElements =
            output->elementSize();


    report
            << "Output elements: "
            << outputElements
            << "\n";


    // --------------------------------------------------------
    // Validate output shape
    // --------------------------------------------------------

    bool correctOutputShape =
            outputShape.size() == 4 &&
            outputShape[0] == 1 &&
            outputShape[1] == 3 &&
            outputShape[2] == 512 &&
            outputShape[3] == 512;


    if (!correctOutputShape) {

        report
                << "Output shape: FAIL\n"
                << "Expected [1, 3, 512, 512]\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Output shape: PASS\n";


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

        report
                << "Output host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    // --------------------------------------------------------
    // Analyze output
    // --------------------------------------------------------

    float minimum =
            std::numeric_limits<float>::infinity();


    float maximum =
            -std::numeric_limits<float>::infinity();


    bool finite = true;


    size_t nanCount = 0;
    size_t infCount = 0;


    for (size_t i = 0;
         i < outputHost.elementSize();
         ++i) {

        float value =
                outputPtr[i];


        if (std::isnan(value)) {

            ++nanCount;

            finite = false;

            continue;
        }


        if (std::isinf(value)) {

            ++infCount;

            finite = false;

            continue;
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


    report
            << "Output[0]: "
            << outputPtr[0]
            << "\n";


    report
            << "NaN count: "
            << nanCount
            << "\n";


    report
            << "Inf count: "
            << infCount
            << "\n";


    if (finite) {

        report
                << "Output min: "
                << minimum
                << "\n";


        report
                << "Output max: "
                << maximum
                << "\n";


        report
                << "Output finite: YES\n";

    } else {

        report
                << "Output min: nan\n"
                << "Output max: nan\n"
                << "Output finite: NO\n";
    }


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    interpreter->releaseSession(
            session);

    delete interpreter;


    report
            << "============================================\n";


    if (finite) {

        report
                << "ZERO LATENT VAE TEST: PASS\n";

    } else {

        report
                << "ZERO LATENT VAE TEST: FAILED NUMERIC OUTPUT\n";
    }


    report
            << "============================================\n";


    return report.str();
}


// ============================================================
// VAE OpenCL TEST
//
// Kept for later testing.
// nativeTestVae currently uses CPU only.
// ============================================================

std::string runVaeOpenClTest(
        const std::string& vaePath) {

    std::ostringstream report;


    report
            << "============================================\n"
            << "SANA VAE OPENCL TEST\n"
            << "============================================\n";


    report
            << "Model:\n"
            << vaePath
            << "\n";


    long long fileSize =
            getFileSize(
                    vaePath);


    if (fileSize <= 0) {

        report
                << "File: FAIL\n";

        return report.str();
    }


    report
            << "File size: "
            << fileSize
            << " bytes\n";


    report
            << "Creating MNN interpreter...\n";


    MNN::Interpreter* interpreter =
            MNN::Interpreter::createFromFile(
                    vaePath.c_str());


    if (interpreter == nullptr) {

        report
                << "Interpreter: FAIL\n";

        return report.str();
    }


    report
            << "Interpreter: PASS\n";


    report
            << "Creating OpenCL session...\n";


    MNN::Session* session =
            createOpenClSession(
                    interpreter);


    if (session == nullptr) {

        report
                << "OpenCL session: FAIL\n";

        delete interpreter;

        return report.str();
    }


    report
            << "OpenCL session: PASS\n";


    MNN::Tensor* input =
            interpreter->getSessionInput(
                    session,
                    "latent");


    if (input == nullptr) {

        report
                << "Input tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::vector<int> inputShape =
            input->shape();


    report
            << "Input shape: "
            << formatShape(
                    inputShape)
            << "\n";


    bool correctShape =
            inputShape.size() == 4 &&
            inputShape[0] == 1 &&
            inputShape[1] == 32 &&
            inputShape[2] == 16 &&
            inputShape[3] == 16;


    if (!correctShape) {

        report
                << "Input shape: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Input shape: PASS\n";


    // --------------------------------------------------------
    // Zero latent
    // --------------------------------------------------------

    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE);


    float* hostPtr =
            hostTensor.host<float>();


    if (hostPtr == nullptr) {

        report
                << "Host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::fill(
            hostPtr,
            hostPtr +
                    hostTensor.elementSize(),
            0.0f);


    report
            << "ZERO LATENT TEST\n"
            << "All latent values set to exactly 0.0\n";


    report
            << "Latent sample: "
            << hostPtr[0]
            << ", "
            << hostPtr[1]
            << ", "
            << hostPtr[2]
            << "\n";


    input->copyFromHostTensor(
            &hostTensor);


    report
            << "Host -> MNN: PASS\n";


    // --------------------------------------------------------
    // Run OpenCL
    // --------------------------------------------------------

    report
            << "Running VAE inference...\n";


    auto inferenceStart =
            std::chrono::steady_clock::now();


    MNN::ErrorCode result =
            interpreter->runSession(
                    session);


    auto inferenceEnd =
            std::chrono::steady_clock::now();


    double inferenceMs =
            std::chrono::duration<
                    double,
                    std::milli>(
                    inferenceEnd -
                    inferenceStart)
                    .count();


    report
            << "Inference time: "
            << inferenceMs
            << " ms\n";


    report
            << "MNN error code: "
            << static_cast<int>(
                    result)
            << "\n";


    if (result != MNN::NO_ERROR) {

        report
                << "Inference: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    report
            << "Inference: PASS\n";


    MNN::Tensor* output =
            interpreter->getSessionOutput(
                    session,
                    nullptr);


    if (output == nullptr) {

        report
                << "Output tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    std::vector<int> outputShape =
            output->shape();


    report
            << "Output shape: "
            << formatShape(
                    outputShape)
            << "\n";


    MNN::Tensor outputHost(
            output,
            MNN::Tensor::CAFFE);


    output->copyToHostTensor(
            &outputHost);


    float* outputPtr =
            outputHost.host<float>();


    if (outputPtr == nullptr) {

        report
                << "Output host tensor: FAIL\n";

        interpreter->releaseSession(
                session);

        delete interpreter;

        return report.str();
    }


    float minimum =
            std::numeric_limits<float>::infinity();


    float maximum =
            -std::numeric_limits<float>::infinity();


    bool finite = true;

    size_t nanCount = 0;
    size_t infCount = 0;


    for (size_t i = 0;
         i < outputHost.elementSize();
         ++i) {

        float value =
                outputPtr[i];


        if (std::isnan(value)) {

            ++nanCount;

            finite = false;

            continue;
        }


        if (std::isinf(value)) {

            ++infCount;

            finite = false;

            continue;
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


    report
            << "Output[0]: "
            << outputPtr[0]
            << "\n";


    report
            << "NaN count: "
            << nanCount
            << "\n";


    report
            << "Inf count: "
            << infCount
            << "\n";


    if (finite) {

        report
                << "Output min: "
                << minimum
                << "\n"
                << "Output max: "
                << maximum
                << "\n"
                << "Output finite: YES\n";

    } else {

        report
                << "Output min: nan\n"
                << "Output max: nan\n"
                << "Output finite: NO\n";
    }


    interpreter->releaseSession(
            session);

    delete interpreter;


    report
            << "============================================\n";


    if (finite) {

        report
                << "ZERO LATENT VAE OPENCL TEST: PASS\n";

    } else {

        report
                << "ZERO LATENT VAE OPENCL TEST: FAILED NUMERIC OUTPUT\n";
    }


    report
            << "============================================\n";


    return report.str();
}

} // namespace


// ============================================================
// nativeInitialize
//
// Current behavior:
// CPU initialization.
// OpenCL preference is intentionally ignored for startup.
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

    std::lock_guard<std::mutex> lock(
            gMutex);


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
                "Unable to open model asset: " +
                assetName;

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
// nativeIsInitialized
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv* /* env */,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    return gInitialized
           ? JNI_TRUE
           : JNI_FALSE;
}


// ============================================================
// nativeGetBackend
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    return env->NewStringUTF(
            gBackend.c_str());
}


// ============================================================
// nativeGetStatus
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    return env->NewStringUTF(
            gStatus.c_str());
}


// ============================================================
// nativeRelease
// ============================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv* /* env */,
        jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(
            gMutex);


    releaseGlobalSession();


    gBackend = "NONE";

    gStatus = "Released";
}


// ============================================================
// nativeTestTransformer
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


    (void) cache;


    std::string result =
            runTransformerCpuTest(
                    path);


    return env->NewStringUTF(
            result.c_str());
}


// ============================================================
// nativeTestVae
//
// CPU ONLY
// ZERO LATENT
// SHA-256 INCLUDED
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


    (void) cache;


    std::string result =
            runVaeCpuTest(
                    path);


    return env->NewStringUTF(
            result.c_str());
}


// ============================================================
// nativeTestModels
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


    (void) cache;


    std::ostringstream report;


    report
            << "============================================\n"
            << "SANA MODEL DIAGNOSTIC\n"
            << "============================================\n\n";


    report
            << runTransformerCpuTest(
                    transformer);


    report
            << "\n\n";


    report
            << runVaeCpuTest(
                    vae);


    report
            << "\n============================================\n";


    return env->NewStringUTF(
            report.str().c_str());
}
