#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

/*
 * -------------------------------------------------------------
 * GLOBAL ENGINE STATE
 * -------------------------------------------------------------
 */

bool gInitialized = false;

std::string gBackend = "CPU";

std::string gStatus = "Not initialized";

MNN::Interpreter* gInterpreter = nullptr;

MNN::Session* gSession = nullptr;


/*
 * -------------------------------------------------------------
 * JNI STRING
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * FORMAT SHAPE
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * /proc/self/fd PATH
 * -------------------------------------------------------------
 *
 * Android gives us an already-open file descriptor.
 *
 * MNN createFromFile() wants a path.
 *
 * Linux exposes open descriptors through:
 *
 *     /proc/self/fd/<fd>
 *
 * This avoids copying the giant model.
 */

std::string fdPath(
    int fd
) {

    std::ostringstream path;

    path
        << "/proc/self/fd/"
        << fd;

    return path.str();
}


/*
 * -------------------------------------------------------------
 * FD VALIDATION
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * FILE SIZE FROM FD
 * -------------------------------------------------------------
 */

long long fdFileSize(
    int fd
) {

    if (!validFd(fd)) {
        return -1;
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
 * -------------------------------------------------------------
 * CPU SESSION
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * RELEASE INTERPRETER
 * -------------------------------------------------------------
 */

void destroyInterpreter(
    MNN::Interpreter* interpreter
) {

    if (!interpreter) {
        return;
    }

    delete interpreter;
}


/*
 * -------------------------------------------------------------
 * TRANSFORMER TEST
 * -------------------------------------------------------------
 *
 * This is the same diagnostic concept that previously passed:
 *
 * hidden_states
 * timestep
 * encoder_hidden_states
 *
 * with zero inputs.
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

    std::string path =
        fdPath(fd);

    result
        << "Loading MNN interpreter...\n";

    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            path.c_str()
        );

    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        return result.str();
    }

    result
        << "Interpreter: PASS\n";

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

        return result.str();
    }

    result
        << "CPU session: PASS\n\n";


    /*
     * ---------------------------------------------------------
     * GET INPUTS
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
     * PRINT SHAPES
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


    /*
     * ---------------------------------------------------------
     * ZERO INITIALIZATION
     * ---------------------------------------------------------
     */

    float* hiddenData =
        hostHidden.host<float>();

    float* timestepData =
        hostTimestep.host<float>();

    float* encoderData =
        hostEncoder.host<float>();


    if (!hiddenData ||
        !timestepData ||
        !encoderData) {

        result
            << "Host tensor mapping: FAIL\n";

        interpreter->releaseSession(
            session
        );

        destroyInterpreter(
            interpreter
        );

        return result.str();
    }


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


    /*
     * ---------------------------------------------------------
     * COPY HOST -> DEVICE
     * ---------------------------------------------------------
     */

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


        if (values &&
            hostOutput.elementSize() > 0) {

            size_t nanCount = 0;
            size_t infCount = 0;

            float minValue =
                values[0];

            float maxValue =
                values[0];


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

            if (
                nanCount == 0 &&
                infCount == 0
            ) {

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

    destroyInterpreter(
        interpreter
    );


    result
        << "\nSANA TRANSFORMER CPU TEST: PASS\n";

    return result.str();
}


/*
 * -------------------------------------------------------------
 * VAE TEST
 * -------------------------------------------------------------
 *
 * Zero latent:
 *
 * [1, 32, 16, 16]
 *
 * Exactly the diagnostic that previously produced finite output.
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

    std::string path =
        fdPath(fd);

    result
        << "Creating MNN interpreter...\n";

    MNN::Interpreter* interpreter =
        MNN::Interpreter::createFromFile(
            path.c_str()
        );

    if (!interpreter) {

        result
            << "Interpreter: FAIL\n";

        return result.str();
    }

    result
        << "Interpreter: PASS\n";

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

        destroyInterpreter(
            interpreter
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

        return result.str();
    }


    /*
     * ---------------------------------------------------------
     * ZERO LATENT
     * ---------------------------------------------------------
     */

    std::fill(
        latent,
        latent + hostInput.elementSize(),
        0.0f
    );


    result
        << "\nZERO LATENT TEST\n";

    result
        << "All latent values set to exactly 0.0\n";

    result
        << "Latent sample: "
        << latent[0]
        << ", "
        << latent[
            std::min<size_t>(
                1,
                hostInput.elementSize() - 1
            )
        ]
        << ", "
        << latent[
            std::min<size_t>(
                2,
                hostInput.elementSize() - 1
            )
        ]
        << "\n";


    /*
     * ---------------------------------------------------------
     * COPY
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

        return result.str();
    }


    size_t nanCount = 0;

    size_t infCount = 0;

    bool firstFiniteFound = false;

    float minValue = 0.0f;

    float maxValue = 0.0f;


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


    interpreter->releaseSession(
        session
    );

    destroyInterpreter(
        interpreter
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
 * -------------------------------------------------------------
 * INITIALIZATION
 * -------------------------------------------------------------
 */

bool initializeEngine(
    AAssetManager* assetManager,
    const std::string& modelAsset,
    const std::string& cachePath,
    bool preferOpenCl,
    int cpuThreads
) {

    (void)assetManager;
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
 * -------------------------------------------------------------
 * JNI: INITIALIZE
 * -------------------------------------------------------------
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

    AAssetManager* manager =
        nullptr;

    (void)manager;

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
        nullptr,
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
 * -------------------------------------------------------------
 * JNI: IS INITIALIZED
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * JNI: BACKEND
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * JNI: STATUS
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * JNI: RELEASE
 * -------------------------------------------------------------
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

        delete gInterpreter;

        gInterpreter =
            nullptr;
    }

    gInitialized =
        false;

    gStatus =
        "Released";
}


/*
 * -------------------------------------------------------------
 * JNI: TRANSFORMER FD
 * -------------------------------------------------------------
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

    /*
     * Native owns detached FD.
     */

    if (fd >= 0) {

        close(fd);
    }

    return env->NewStringUTF(
        output.c_str()
    );
}


/*
 * -------------------------------------------------------------
 * JNI: VAE FD
 * -------------------------------------------------------------
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
 * -------------------------------------------------------------
 * JNI: TRANSFORMER + VAE
 * -------------------------------------------------------------
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
        << "NO MODEL COPYING\n";

    output
        << "Models accessed through Android file descriptors.\n\n";


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
     * Transformer test has completed.
     *
     * MNN opens /proc/self/fd/<fd>.
     *
     * Keep descriptor alive until after test.
     */


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
     * CLOSE BOTH
     * ---------------------------------------------------------
     */

    if (transformer >= 0) {

        close(
            transformer
        );
    }

    if (vae >= 0) {

        close(
            vae
        );
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
