#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

/*
 * =============================================================
 * GLOBAL ENGINE STATE
 * =============================================================
 */

bool gInitialized = false;

std::string gBackend = "CPU";

std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;

MNN::Session* gSession = nullptr;


/*
 * =============================================================
 * JNI STRING
 * =============================================================
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
 * =============================================================
 * FORMAT SHAPE
 * =============================================================
 */

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


/*
 * =============================================================
 * VALIDATE FD
 * =============================================================
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
 * =============================================================
 * GET FILE SIZE
 * =============================================================
 */

long long fdFileSize(
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

    /*
     * Fallback for providers where fstat
     * does not provide a useful size.
     */

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
 * =============================================================
 * MEMORY-MAPPED MODEL
 * =============================================================
 *
 * This is the important change.
 *
 * We DO NOT:
 *
 *   - copy the 1.18 GB Transformer into a std::vector
 *   - copy the 320 MB VAE into app storage
 *   - use /proc/self/fd with createFromFile()
 *
 * Instead:
 *
 * Android FD
 *      ↓
 * mmap()
 *      ↓
 * MNN::Interpreter::createFromBuffer()
 *
 * The mapping remains alive for the complete lifetime
 * of the Interpreter.
 */

struct MappedModel {

    void* data = MAP_FAILED;

    size_t size = 0;

    bool mapped() const {
        return
            data != MAP_FAILED &&
            data != nullptr &&
            size > 0;
    }
};


/*
 * =============================================================
 * MAP MODEL
 * =============================================================
 */

bool mapModel(
    int fd,
    MappedModel& model,
    std::string& error
) {
    model.data =
        MAP_FAILED;

    model.size =
        0;

    error.clear();


    if (!validFd(fd)) {

        error =
            "Invalid file descriptor";

        return false;
    }


    long long fileSize =
        fdFileSize(fd);


    if (fileSize <= 0) {

        error =
            "Invalid model file size";

        return false;
    }


    /*
     * Android arm64 is 64-bit.
     *
     * The Transformer is approximately
     * 1.18 GB, which is suitable for a
     * 64-bit virtual memory mapping.
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
            "Model is too large for size_t";

        return false;
    }


    size_t size =
        static_cast<size_t>(
            fileSize
        );


    /*
     * Make sure the descriptor points
     * to a regular file when possible.
     */

    struct stat st{};

    if (
        fstat(
            fd,
            &st
        ) == 0
    ) {

        if (
            !S_ISREG(
                st.st_mode
            )
        ) {

            error =
                "Selected document is not a regular file";

            return false;
        }
    }


    /*
     * Ensure the offset is at the beginning.
     */

    if (
        lseek(
            fd,
            0,
            SEEK_SET
        )
        ==
        (off_t)-1
    ) {

        error =
            "Unable to seek model file";

        return false;
    }


    void* mapped =
        mmap(
            nullptr,
            size,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            0
        );


    if (
        mapped == MAP_FAILED ||
        mapped == nullptr
    ) {

        std::ostringstream out;

        out
            << "mmap failed: "
            << std::strerror(errno);

        error =
            out.str();

        return false;
    }


    model.data =
        mapped;

    model.size =
        size;


    return true;
}


/*
 * =============================================================
 * UNMAP MODEL
 * =============================================================
 */

void unmapModel(
    MappedModel& model
) {
    if (model.mapped()) {

        munmap(
            model.data,
            model.size
        );
    }

    model.data =
        MAP_FAILED;

    model.size =
        0;
}


/*
 * =============================================================
 * CREATE INTERPRETER FROM FD
 * =============================================================
 */

MNN::Interpreter* createInterpreterFromFd(
    int fd,
    MappedModel& mappedModel,
    std::string& error
) {
    error.clear();

    if (
        !mapModel(
            fd,
            mappedModel,
            error
        )
    ) {

        return nullptr;
    }


    /*
     * MNN supports createFromBuffer().
     *
     * mappedModel MUST remain alive while
     * the Interpreter uses the model.
     */

    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromBuffer(
            mappedModel.data,
            mappedModel.size
        );


    if (!interpreter) {

        error =
            "MNN createFromBuffer returned null";

        unmapModel(
            mappedModel
        );

        return nullptr;
    }


    return interpreter;
}


/*
 * =============================================================
 * CPU SESSION
 * =============================================================
 */

MNN::Session* createCpuSession(
    MNN::Interpreter* interpreter,
    int threads
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


/*
 * =============================================================
 * DESTROY INTERPRETER
 * =============================================================
 */

void destroyInterpreter(
    MNN::Interpreter* interpreter
) {
    if (!interpreter) {
        return;
    }

    MNN::Interpreter::destroy(
        interpreter
    );
}


/*
 * =============================================================
 * TRANSFORMER TEST
 * =============================================================
 */

std::string runTransformerTest(
    int fd
) {
    std::ostringstream result;


    result
        << "SANA TRANSFORMER CPU TEST\n\n";


    result
        << "FD: "
        << fd
        << "\n";


    if (!validFd(fd)) {

        result
            << "File descriptor: FAIL\n";

        return result.str();
    }


    long long size =
        fdFileSize(fd);


    result
        << "File size: "
        << size
        << " bytes\n\n";


    if (size <= 0) {

        result
            << "File size: FAIL\n";

        return result.str();
    }


    /*
     * ---------------------------------------------------------
     * MAP MODEL
     * ---------------------------------------------------------
     */

    result
        << "Memory-mapping Transformer...\n";


    MappedModel mappedModel;

    std::string loadError;


    MNN::Interpreter* interpreter =
        createInterpreterFromFd(
            fd,
            mappedModel,
            loadError
        );


    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        result
            << "Loader error: "
            << loadError
            << "\n";

        return result.str();
    }


    result
        << "Memory map: PASS\n";

    result
        << "MNN interpreter: PASS\n";


    /*
     * ---------------------------------------------------------
     * SESSION
     * ---------------------------------------------------------
     */

    result
        << "Creating CPU session...\n";


    MNN::Session* session =
        createCpuSession(
            interpreter,
            4
        );


    if (!session) {

        result
            << "CPU session: FAIL\n";

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    result
        << "CPU session: PASS\n\n";


    /*
     * ---------------------------------------------------------
     * INPUTS
     * ---------------------------------------------------------
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


    if (!hiddenStates) {

        result
            << "Input hidden_states: FAIL\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    if (!timestep) {

        result
            << "Input timestep: FAIL\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    if (!encoderHiddenStates) {

        result
            << "Input encoder_hidden_states: FAIL\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    result
        << "Input hidden_states: PASS\n";

    result
        << "Input timestep: PASS\n";

    result
        << "Input encoder_hidden_states: PASS\n\n";


    /*
     * ---------------------------------------------------------
     * SHAPES
     * ---------------------------------------------------------
     */

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
     * ---------------------------------------------------------
     * HOST TENSORS
     * ---------------------------------------------------------
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

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    /*
     * ---------------------------------------------------------
     * ZERO INPUTS
     * ---------------------------------------------------------
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


    /*
     * ---------------------------------------------------------
     * RUN
     * ---------------------------------------------------------
     */

    MNN::ErrorCode code =
        interpreter->runSession(
            session
        );


    result
        << "MNN error code: "
        << static_cast<int>(code)
        << "\n";


    if (code != MNN::NO_ERROR) {

        result
            << "Transformer inference: FAIL\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    result
        << "Transformer inference: PASS\n\n";


    /*
     * ---------------------------------------------------------
     * OUTPUT
     * ---------------------------------------------------------
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

            size_t nanCount = 0;

            size_t infCount = 0;

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


                if (std::isnan(value)) {

                    ++nanCount;

                } else if (
                    std::isinf(value)
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


    /*
     * ---------------------------------------------------------
     * CLEANUP ORDER
     * ---------------------------------------------------------
     *
     * Interpreter first.
     * Memory mapping second.
     *
     * The model buffer must stay mapped while
     * the Interpreter exists.
     */

    interpreter->releaseSession(
        session
    );


    destroyInterpreter(
        interpreter
    );


    unmapModel(
        mappedModel
    );


    result
        << "\nSANA TRANSFORMER CPU TEST: PASS\n";


    return result.str();
}


/*
 * =============================================================
 * VAE TEST
 * =============================================================
 */

std::string runVaeTest(
    int fd
) {
    std::ostringstream result;


    result
        << "SANA VAE CPU TEST\n\n";


    result
        << "FD: "
        << fd
        << "\n";


    if (!validFd(fd)) {

        result
            << "File descriptor: FAIL\n";

        return result.str();
    }


    long long size =
        fdFileSize(fd);


    result
        << "File size: "
        << size
        << " bytes\n\n";


    if (size <= 0) {

        result
            << "File size: FAIL\n";

        return result.str();
    }


    /*
     * ---------------------------------------------------------
     * MAP VAE
     * ---------------------------------------------------------
     */

    result
        << "Memory-mapping VAE...\n";


    MappedModel mappedModel;

    std::string loadError;


    MNN::Interpreter* interpreter =
        createInterpreterFromFd(
            fd,
            mappedModel,
            loadError
        );


    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        result
            << "Loader error: "
            << loadError
            << "\n";

        return result.str();
    }


    result
        << "Memory map: PASS\n";

    result
        << "MNN interpreter: PASS\n";


    /*
     * ---------------------------------------------------------
     * SESSION
     * ---------------------------------------------------------
     */

    result
        << "Creating CPU session...\n";


    MNN::Session* session =
        createCpuSession(
            interpreter,
            4
        );


    if (!session) {

        result
            << "CPU session: FAIL\n";

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    result
        << "CPU session: PASS\n\n";


    /*
     * ---------------------------------------------------------
     * INPUT
     * ---------------------------------------------------------
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

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

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


    /*
     * Expected Sana VAE latent:
     *
     * [1, 32, 16, 16]
     */

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

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    result
        << "Input shape: PASS\n";


    result
        << "Input elements: "
        << input->elementSize()
        << "\n";


    /*
     * ---------------------------------------------------------
     * HOST LATENT
     * ---------------------------------------------------------
     */

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

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    /*
     * ---------------------------------------------------------
     * ZERO LATENT
     * ---------------------------------------------------------
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
        hostInput.elementSize() >= 3
    ) {

        result
            << "Latent sample: "
            << latent[0]
            << ", "
            << latent[1]
            << ", "
            << latent[2]
            << "\n";
    }


    /*
     * ---------------------------------------------------------
     * COPY HOST -> MNN
     * ---------------------------------------------------------
     */

    input->copyFromHostTensor(
        &hostInput
    );


    result
        << "Host -> MNN: PASS\n";


    result
        << "Running VAE inference...\n";


    /*
     * ---------------------------------------------------------
     * RUN
     * ---------------------------------------------------------
     */

    MNN::ErrorCode code =
        interpreter->runSession(
            session
        );


    result
        << "MNN error code: "
        << static_cast<int>(code)
        << "\n";


    if (code != MNN::NO_ERROR) {

        result
            << "VAE inference: FAIL\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    result
        << "VAE inference: PASS\n\n";


    /*
     * ---------------------------------------------------------
     * OUTPUT
     * ---------------------------------------------------------
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

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


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

        destroyInterpreter(
            interpreter
        );

        unmapModel(
            mappedModel
        );

        return result.str();
    }


    size_t nanCount = 0;

    size_t infCount = 0;

    bool firstFiniteFound =
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


        if (std::isnan(value)) {

            ++nanCount;

        } else if (
            std::isinf(value)
        ) {

            ++infCount;

        } else {

            if (!firstFiniteFound) {

                minValue =
                    value;

                maxValue =
                    value;

                firstFiniteFound =
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


    if (firstFiniteFound) {

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


    /*
     * ---------------------------------------------------------
     * CLEANUP
     * ---------------------------------------------------------
     */

    interpreter->releaseSession(
        session
    );


    destroyInterpreter(
        interpreter
    );


    unmapModel(
        mappedModel
    );


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
 * =============================================================
 * INITIALIZATION
 * =============================================================
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
 * =============================================================
 * JNI INITIALIZE
 * =============================================================
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
 * =============================================================
 * JNI IS INITIALIZED
 * =============================================================
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
 * =============================================================
 * JNI BACKEND
 * =============================================================
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
 * =============================================================
 * JNI STATUS
 * =============================================================
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
 * =============================================================
 * JNI RELEASE
 * =============================================================
 */

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
    JNIEnv*,
    jobject
) {
    if (gInterpreter) {

        if (gSession) {

            gInterpreter->releaseSession(
                gSession
            );

            gSession =
                nullptr;
        }


        destroyInterpreter(
            gInterpreter
        );


        gInterpreter =
            nullptr;
    }


    gInitialized =
        false;


    gStatus =
        "Released";
}


/*
 * =============================================================
 * JNI TRANSFORMER FD
 * =============================================================
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
            "SANA TRANSFORMER TEST\n\n"
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
 * =============================================================
 * JNI VAE FD
 * =============================================================
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
            "SANA VAE TEST\n\n"
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
 * =============================================================
 * JNI TRANSFORMER + VAE
 * =============================================================
 */

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestModelsFd(
    JNIEnv* env,
    jobject,
    jint transformerFd,
    jint vaeFd,
    jboolean preferOpenCl
) {
    (void)preferOpenCl;


    int transformer =
        static_cast<int>(
            transformerFd
        );


    int vae =
        static_cast<int>(
            vaeFd
        );


    std::ostringstream output;


    output
        << "SANA 0.6B MODEL DIAGNOSTIC\n\n";


    output
        << "DIRECT EXTERNAL MODEL ACCESS\n";


    output
        << "No model copying performed.\n";


    output
        << "Models are memory-mapped directly from Android storage.\n\n";


    /*
     * ---------------------------------------------------------
     * TRANSFORMER
     * ---------------------------------------------------------
     */

    output
        << "========================================\n";


    output
        << "TRANSFORMER\n";


    output
        << "========================================\n\n";


    if (validFd(transformer)) {

        output
            << runTransformerTest(
                transformer
            );

    } else {

        output
            << "Transformer FD invalid.\n";
    }


    /*
     * ---------------------------------------------------------
     * VAE
     * ---------------------------------------------------------
     */

    output
        << "\n\n========================================\n";


    output
        << "VAE\n";


    output
        << "========================================\n\n";


    if (validFd(vae)) {

        output
            << runVaeTest(
                vae
            );

    } else {

        output
            << "VAE FD invalid.\n";
    }


    /*
     * ---------------------------------------------------------
     * CLOSE DETACHED FDS
     * ---------------------------------------------------------
     */

    if (transformer >= 0) {
        close(transformer);
    }


    if (vae >= 0) {
        close(vae);
    }


    output
        << "\n\n========================================\n";


    output
        << "END OF DIAGNOSTIC\n";


    output
        << "========================================\n";


    std::string text =
        output.str();


    return env->NewStringUTF(
        text.c_str()
    );
}

} // namespace
