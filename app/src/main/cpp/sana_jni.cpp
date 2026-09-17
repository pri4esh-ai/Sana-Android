#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

bool gInitialized = false;

std::string gBackend = "CPU";

std::string gStatus = "Not initialized";


/*
 * ============================================================
 * JNI STRING
 * ============================================================
 */

std::string jstringToString(
    JNIEnv* env,
    jstring value
) {
    if (!value) {
        return "";
    }

    const char* chars =
        env->GetStringUTFChars(
            value,
            nullptr
        );

    std::string result =
        chars ? chars : "";

    if (chars) {
        env->ReleaseStringUTFChars(
            value,
            chars
        );
    }

    return result;
}


/*
 * ============================================================
 * FD VALIDATION
 * ============================================================
 */

bool validFd(
    int fd
) {
    if (fd < 0) {
        return false;
    }

    return fcntl(
        fd,
        F_GETFD
    ) != -1;
}


/*
 * ============================================================
 * FILE SIZE
 * ============================================================
 */

long long getFileSize(
    int fd
) {
    if (!validFd(fd)) {
        return -1;
    }

    struct stat st{};

    if (fstat(fd, &st) == 0) {
        if (st.st_size > 0) {
            return static_cast<long long>(
                st.st_size
            );
        }
    }

    off_t current =
        lseek(
            fd,
            0,
            SEEK_CUR
        );

    if (current == (off_t)-1) {
        current = 0;
    }

    off_t end =
        lseek(
            fd,
            0,
            SEEK_END
        );

    if (end == (off_t)-1) {
        return -1;
    }

    lseek(
        fd,
        current,
        SEEK_SET
    );

    return static_cast<long long>(
        end
    );
}


/*
 * ============================================================
 * READ EXACTLY
 * ============================================================
 */

bool readFully(
    int fd,
    uint8_t* buffer,
    size_t size
) {
    if (!validFd(fd)) {
        return false;
    }

    /*
     * Always start at the beginning.
     */
    if (lseek(fd, 0, SEEK_SET) == (off_t)-1) {
        /*
         * Some SAF descriptors may not support seek.
         * In that case continue from the current position.
         */
    }

    size_t total = 0;

    while (total < size) {

        ssize_t count =
            read(
                fd,
                buffer + total,
                size - total
            );

        if (count < 0) {

            if (errno == EINTR) {
                continue;
            }

            return false;
        }

        if (count == 0) {
            break;
        }

        total +=
            static_cast<size_t>(
                count
            );
    }

    return total == size;
}


/*
 * ============================================================
 * MNN BUFFER LOADER
 *
 * MNN officially supports createFromBuffer().
 *
 * The Android file itself remains external.
 * We only create a temporary native buffer for MNN.
 * ============================================================
 */

MNN::Interpreter* loadModelFromFd(
    int fd,
    std::vector<uint8_t>& modelBuffer,
    std::string& error
) {
    if (!validFd(fd)) {

        error =
            "Invalid file descriptor";

        return nullptr;
    }

    long long fileSize =
        getFileSize(fd);

    if (fileSize <= 0) {

        error =
            "Unable to determine model file size";

        return nullptr;
    }

    /*
     * Protect against impossible/overflowing sizes.
     */
    if (
        static_cast<unsigned long long>(
            fileSize
        )
        >
        static_cast<unsigned long long>(
            SIZE_MAX
        )
    ) {

        error =
            "Model file is too large for address space";

        return nullptr;
    }

    size_t size =
        static_cast<size_t>(
            fileSize
        );

    /*
     * Reserve the exact model size.
     */
    try {

        modelBuffer.resize(
            size
        );

    } catch (...) {

        error =
            "Native memory allocation failed for model buffer";

        modelBuffer.clear();

        return nullptr;
    }

    /*
     * Read the complete MNN file.
     */
    if (!readFully(
            fd,
            modelBuffer.data(),
            modelBuffer.size()
        )) {

        error =
            "Unable to read complete model from file descriptor";

        modelBuffer.clear();

        return nullptr;
    }

    /*
     * IMPORTANT:
     *
     * createFromBuffer() keeps/uses the supplied model buffer
     * while the Interpreter is alive.
     *
     * Therefore modelBuffer MUST remain alive until the
     * Interpreter has been destroyed.
     */
    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromBuffer(
            modelBuffer.data(),
            modelBuffer.size()
        );

    if (!interpreter) {

        error =
            "MNN createFromBuffer() returned null";

        modelBuffer.clear();

        return nullptr;
    }

    return interpreter;
}


/*
 * ============================================================
 * SHAPE FORMAT
 * ============================================================
 */

std::string formatShape(
    const std::vector<int>& shape
) {
    std::ostringstream out;

    out << "[";

    for (
        size_t i = 0;
        i < shape.size();
        ++i
    ) {

        if (i != 0) {
            out << ", ";
        }

        out << shape[i];
    }

    out << "]";

    return out.str();
}


/*
 * ============================================================
 * CPU SESSION
 * ============================================================
 */

MNN::Session* createCpuSession(
    MNN::Interpreter* interpreter
) {
    if (!interpreter) {
        return nullptr;
    }

    MNN::ScheduleConfig config;

    config.type =
        MNN_FORWARD_CPU;

    config.numThread =
        4;

    /*
     * High precision for diagnostic.
     */
    MNN::BackendConfig backendConfig;

    backendConfig.precision =
        MNN::BackendConfig::Precision_High;

    config.backendConfig =
        &backendConfig;

    return interpreter->createSession(
        config
    );
}


/*
 * ============================================================
 * TRANSFORMER TEST
 * ============================================================
 */

std::string runTransformerTest(
    int fd
) {
    std::ostringstream result;

    result
        << "SANA TRANSFORMER CPU TEST\n\n";

    result
        << "File descriptor: "
        << fd
        << "\n";


    if (!validFd(fd)) {

        result
            << "File descriptor: FAIL\n";

        return result.str();
    }


    long long fileSize =
        getFileSize(fd);

    result
        << "File size: "
        << fileSize
        << " bytes\n\n";


    if (fileSize <= 0) {

        result
            << "File size: FAIL\n";

        return result.str();
    }


    result
        << "Loading model through MNN buffer API...\n";


    std::vector<uint8_t> modelBuffer;

    std::string loadError;


    MNN::Interpreter* interpreter =
        loadModelFromFd(
            fd,
            modelBuffer,
            loadError
        );


    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        result
            << "Reason: "
            << loadError
            << "\n";

        return result.str();
    }


    result
        << "Interpreter: PASS\n";


    result
        << "Model loaded with createFromBuffer()\n";


    result
        << "Creating CPU session...\n";


    MNN::Session* session =
        createCpuSession(
            interpreter
        );


    if (!session) {

        result
            << "CPU session: FAIL\n";

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "CPU session: PASS\n\n";


    /*
     * ========================================================
     * FIND INPUTS
     * ========================================================
     */

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


    /*
     * Some converted MNN models can have different names.
     *
     * If named lookup fails, inspect all inputs.
     */
    if (
        !hiddenStates ||
        !timestep ||
        !encoderHiddenStates
    ) {

        result
            << "Named input lookup incomplete.\n";

        result
            << "Attempting all-input diagnostic...\n";


        auto allInputs =
            interpreter->getSessionInputAll(
                session
            );


        result
            << "Input count: "
            << allInputs.size()
            << "\n";


        for (const auto& item : allInputs) {

            result
                << "Input: "
                << item.first
                << " shape="
                << formatShape(
                    item.second->shape()
                )
                << "\n";
        }


        result
            << "\nTransformer input lookup: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "hidden_states: PASS\n";

    result
        << "timestep: PASS\n";

    result
        << "encoder_hidden_states: PASS\n\n";


    result
        << "hidden_states shape: "
        << formatShape(
            hiddenStates->shape()
        )
        << "\n";

    result
        << "timestep shape: "
        << formatShape(
            timestep->shape()
        )
        << "\n";

    result
        << "encoder_hidden_states shape: "
        << formatShape(
            encoderHiddenStates->shape()
        )
        << "\n\n";


    /*
     * ========================================================
     * HOST TENSORS
     * ========================================================
     */

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


    float* hiddenData =
        hostHidden.host<float>();

    float* timestepData =
        hostTimestep.host<float>();

    float* encoderData =
        hostEncoder.host<float>();


    if (
        !hiddenData ||
        !timestepData ||
        !encoderData
    ) {

        result
            << "Host tensor mapping: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    /*
     * Zero diagnostic input.
     */

    std::fill(
        hiddenData,
        hiddenData +
            hostHidden.elementSize(),
        0.0f
    );

    std::fill(
        timestepData,
        timestepData +
            hostTimestep.elementSize(),
        0.0f
    );

    std::fill(
        encoderData,
        encoderData +
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


    result
        << "Zero inputs: PASS\n";

    result
        << "Running Transformer...\n";


    MNN::ErrorCode code =
        interpreter->runSession(
            session
        );


    result
        << "MNN error code: "
        << static_cast<int>(
            code
        )
        << "\n";


    if (code != MNN::NO_ERROR) {

        result
            << "Transformer inference: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "Transformer inference: PASS\n\n";


    /*
     * ========================================================
     * OUTPUT
     * ========================================================
     */

    MNN::Tensor* output =
        interpreter->getSessionOutput(
            session,
            "sample"
        );


    if (!output) {

        output =
            interpreter->getSessionOutput(
                session,
                nullptr
            );
    }


    if (!output) {

        result
            << "Output tensor: FAIL\n";

    } else {

        result
            << "Output tensor: PASS\n";

        result
            << "Output shape: "
            << formatShape(
                output->shape()
            )
            << "\n";

        result
            << "Output elements: "
            << output->elementSize()
            << "\n";


        MNN::Tensor hostOutput(
            output,
            MNN::Tensor::CAFFE
        );


        const float* values =
            hostOutput.host<float>();


        if (
            values &&
            hostOutput.elementSize() > 0
        ) {

            size_t nanCount =
                0;

            size_t infCount =
                0;

            bool finiteFound =
                false;

            float minValue =
                0.0f;

            float maxValue =
                0.0f;


            for (
                size_t i = 0;
                i < hostOutput.elementSize();
                ++i
            ) {

                float value =
                    values[i];


                if (
                    std::isnan(
                        value
                    )
                ) {

                    ++nanCount;

                } else if (
                    std::isinf(
                        value
                    )
                ) {

                    ++infCount;

                } else {

                    if (!finiteFound) {

                        minValue =
                            value;

                        maxValue =
                            value;

                        finiteFound =
                            true;

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
            }


            result
                << "NaN count: "
                << nanCount
                << "\n";

            result
                << "Inf count: "
                << infCount
                << "\n";

            result
                << "Output finite: "
                << (
                    nanCount == 0 &&
                    infCount == 0
                        ? "YES"
                        : "NO"
                )
                << "\n";


            if (finiteFound) {

                result
                    << "Output min: "
                    << minValue
                    << "\n";

                result
                    << "Output max: "
                    << maxValue
                    << "\n";
            }
        }
    }


    interpreter->releaseSession(
        session
    );

    MNN::Interpreter::destroy(
        interpreter
    );

    /*
     * Interpreter is now gone.
     * Buffer can safely be released.
     */
    modelBuffer.clear();


    result
        << "\nSANA TRANSFORMER CPU TEST: PASS\n";


    return result.str();
}


/*
 * ============================================================
 * VAE TEST
 * ============================================================
 */

std::string runVaeTest(
    int fd
) {
    std::ostringstream result;

    result
        << "SANA VAE CPU TEST\n\n";

    result
        << "File descriptor: "
        << fd
        << "\n";


    if (!validFd(fd)) {

        result
            << "File descriptor: FAIL\n";

        return result.str();
    }


    long long fileSize =
        getFileSize(fd);


    result
        << "File size: "
        << fileSize
        << " bytes\n\n";


    if (fileSize <= 0) {

        result
            << "File size: FAIL\n";

        return result.str();
    }


    result
        << "Loading model through MNN buffer API...\n";


    std::vector<uint8_t> modelBuffer;

    std::string loadError;


    MNN::Interpreter* interpreter =
        loadModelFromFd(
            fd,
            modelBuffer,
            loadError
        );


    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        result
            << "Reason: "
            << loadError
            << "\n";

        return result.str();
    }


    result
        << "Interpreter: PASS\n";

    result
        << "Model loaded with createFromBuffer()\n";


    result
        << "Creating CPU session...\n";


    MNN::Session* session =
        createCpuSession(
            interpreter
        );


    if (!session) {

        result
            << "CPU session: FAIL\n";

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "CPU session: PASS\n\n";


    /*
     * ========================================================
     * VAE INPUT
     * ========================================================
     */

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

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "Input shape: "
        << formatShape(
            input->shape()
        )
        << "\n";


    const std::vector<int>& shape =
        input->shape();


    if (
        shape.size() != 4 ||
        shape[0] != 1 ||
        shape[1] != 32 ||
        shape[2] != 16 ||
        shape[3] != 16
    ) {

        result
            << "Expected shape: [1, 32, 16, 16]\n";

        result
            << "Input shape: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "Input shape: PASS\n";


    result
        << "Input elements: "
        << input->elementSize()
        << "\n";


    MNN::Tensor hostInput(
        input,
        MNN::Tensor::CAFFE
    );


    float* latent =
        hostInput.host<float>();


    if (!latent) {

        result
            << "Host tensor: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    /*
     * Zero latent.
     */

    std::fill(
        latent,
        latent +
            hostInput.elementSize(),
        0.0f
    );


    result
        << "\nZERO LATENT TEST\n";

    result
        << "All latent values set to exactly 0.0\n";


    if (
        hostInput.elementSize() > 0
    ) {

        result
            << "Latent first value: "
            << latent[0]
            << "\n";
    }


    input->copyFromHostTensor(
        &hostInput
    );


    result
        << "Host -> MNN: PASS\n";

    result
        << "Running VAE inference...\n";


    MNN::ErrorCode code =
        interpreter->runSession(
            session
        );


    result
        << "MNN error code: "
        << static_cast<int>(
            code
        )
        << "\n";


    if (code != MNN::NO_ERROR) {

        result
            << "VAE inference: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "VAE inference: PASS\n\n";


    /*
     * ========================================================
     * OUTPUT
     * ========================================================
     */

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

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    result
        << "Output tensor: PASS\n";

    result
        << "Output shape: "
        << formatShape(
            output->shape()
        )
        << "\n";

    result
        << "Output elements: "
        << output->elementSize()
        << "\n";


    MNN::Tensor hostOutput(
        output,
        MNN::Tensor::CAFFE
    );


    const float* outputData =
        hostOutput.host<float>();


    if (!outputData) {

        result
            << "Output host mapping: FAIL\n";


        interpreter->releaseSession(
            session
        );

        MNN::Interpreter::destroy(
            interpreter
        );

        modelBuffer.clear();

        return result.str();
    }


    size_t nanCount =
        0;

    size_t infCount =
        0;

    bool finiteFound =
        false;

    float minValue =
        0.0f;

    float maxValue =
        0.0f;


    for (
        size_t i = 0;
        i < hostOutput.elementSize();
        ++i
    ) {

        float value =
            outputData[i];


        if (
            std::isnan(
                value
            )
        ) {

            ++nanCount;

        } else if (
            std::isinf(
                value
            )
        ) {

            ++infCount;

        } else {

            if (!finiteFound) {

                minValue =
                    value;

                maxValue =
                    value;

                finiteFound =
                    true;

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
    }


    result
        << "NaN count: "
        << nanCount
        << "\n";

    result
        << "Inf count: "
        << infCount
        << "\n";


    if (finiteFound) {

        result
            << "Output min: "
            << minValue
            << "\n";

        result
            << "Output max: "
            << maxValue
            << "\n";
    }


    bool finite =
        nanCount == 0 &&
        infCount == 0;


    result
        << "Output finite: "
        << (
            finite
                ? "YES"
                : "NO"
        )
        << "\n";


    interpreter->releaseSession(
        session
    );

    MNN::Interpreter::destroy(
        interpreter
    );

    modelBuffer.clear();


    if (finite) {

        result
            << "\nZERO LATENT VAE TEST: PASS\n";

    } else {

        result
            << "\nZERO LATENT VAE TEST: FAILED NUMERIC OUTPUT\n";
    }


    return result.str();
}


/*
 * ============================================================
 * ENGINE INITIALIZATION
 * ============================================================
 */

bool initializeEngine(
    const std::string& modelAsset,
    const std::string& cachePath,
    bool preferOpenCl,
    int cpuThreads
) {
    (void)modelAsset;
    (void)cachePath;
    (void)preferOpenCl;
    (void)cpuThreads;

    gInitialized =
        true;

    gBackend =
        "CPU";

    gStatus =
        "Initialized";

    return true;
}


/*
 * ============================================================
 * nativeInitialize
 * ============================================================
 */

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
    (void)assetManager;

    std::string asset =
        jstringToString(
            env,
            modelAsset
        );

    std::string cache =
        jstringToString(
            env,
            cachePath
        );

    return initializeEngine(
        asset,
        cache,
        preferOpenCl == JNI_TRUE,
        static_cast<int>(
            cpuThreads
        )
    )
        ? JNI_TRUE
        : JNI_FALSE;
}


/*
 * ============================================================
 * nativeIsInitialized
 * ============================================================
 */

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
    JNIEnv*,
    jobject
) {
    return gInitialized
        ? JNI_TRUE
        : JNI_FALSE;
}


/*
 * ============================================================
 * nativeGetBackend
 * ============================================================
 */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
    JNIEnv* env,
    jobject
) {
    return env->NewStringUTF(
        gBackend.c_str()
    );
}


/*
 * ============================================================
 * nativeGetStatus
 * ============================================================
 */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
    JNIEnv* env,
    jobject
) {
    return env->NewStringUTF(
        gStatus.c_str()
    );
}


/*
 * ============================================================
 * nativeRelease
 * ============================================================
 */

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
    JNIEnv*,
    jobject
) {
    gInitialized =
        false;

    gStatus =
        "Released";
}


/*
 * ============================================================
 * nativeTestTransformerFd
 * ============================================================
 */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformerFd(
    JNIEnv* env,
    jobject,
    jint transformerFd,
    jboolean preferOpenCl
) {
    (void)preferOpenCl;

    int fd =
        static_cast<int>(
            transformerFd
        );


    std::string output;


    if (validFd(fd)) {

        output =
            runTransformerTest(
                fd
            );

    } else {

        output =
            "SANA TRANSFORMER CPU TEST\n\n"
            "Invalid file descriptor.";
    }


    if (fd >= 0) {
        close(fd);
    }


    return env->NewStringUTF(
        output.c_str()
    );
}


/*
 * ============================================================
 * nativeTestVaeFd
 * ============================================================
 */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVaeFd(
    JNIEnv* env,
    jobject,
    jint vaeFd,
    jboolean preferOpenCl
) {
    (void)preferOpenCl;

    int fd =
        static_cast<int>(
            vaeFd
        );


    std::string output;


    if (validFd(fd)) {

        output =
            runVaeTest(
                fd
            );

    } else {

        output =
            "SANA VAE CPU TEST\n\n"
            "Invalid file descriptor.";
    }


    if
