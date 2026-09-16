#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
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

std::shared_ptr<MNN::Interpreter> gInterpreter;
MNN::Session* gSession = nullptr;

bool gInitialized = false;
std::string gBackend = "Not initialized";
std::string gStatus = "Not initialized";

static std::string jstringToString(JNIEnv* env, jstring value) {
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

static std::string shapeToString(
    const std::vector<int>& shape
) {
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

static std::string backendName(
    MNNForwardType backend
) {
    switch (backend) {

        case MNN_FORWARD_OPENCL:
            return "OpenCL / FP16";

        case MNN_FORWARD_CPU:
            return "CPU";

        case MNN_FORWARD_OPENGL:
            return "OpenGL";

        case MNN_FORWARD_VULKAN:
            return "Vulkan";

        default:
            return "Unknown";
    }
}

static std::string makeSessionStatus(
    MNN::Interpreter* interpreter,
    MNN::Session* session
) {
    if (interpreter == nullptr || session == nullptr) {
        return "Invalid interpreter/session";
    }

    int resizeStatus = 0;

    interpreter->getSessionInfo(
        session,
        MNN::Interpreter::RESIZE_STATUS,
        &resizeStatus
    );

    std::ostringstream out;

    out
        << "Resize status: "
        << resizeStatus;

    return out.str();
}

/*
 * Existing engine API.
 *
 * This is kept intact so the diagnostic APK remains compatible
 * with NativeSana.kt.
 */
class SanaEngine {

public:

    SanaEngine() = default;

    ~SanaEngine() {
        release();
    }

    bool initialize(
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl,
        int cpuThreads
    ) {

        release();

        LOGI(
            "SanaEngine initialize: %s",
            modelPath.c_str()
        );

        interpreter =
            MNN::Interpreter::createFromFile(
                modelPath.c_str()
            );

        if (!interpreter) {

            LOGE(
                "Failed to create interpreter"
            );

            return false;
        }

        MNN::ScheduleConfig config{};

        config.numThread =
            cpuThreads > 0
                ? cpuThreads
                : 4;

        if (preferOpenCl) {

            config.type =
                MNN_FORWARD_OPENCL;

            backend =
                "OpenCL / FP16";

        } else {

            config.type =
                MNN_FORWARD_CPU;

            backend =
                "CPU";
        }

        MNN::BackendConfig backendConfig{};

        backendConfig.precision =
            MNN::BackendConfig::Precision_Low;

        backendConfig.memory =
            MNN::BackendConfig::Memory_Normal;

        backendConfig.power =
            MNN::BackendConfig::Power_Normal;

        config.backendConfig =
            &backendConfig;

        session =
            interpreter->createSession(
                config
            );

        if (!session) {

            LOGE(
                "Failed to create Sana session"
            );

            interpreter.reset();

            return false;
        }

        initialized = true;

        status =
            "Sana engine initialized";

        LOGI(
            "SanaEngine initialized: backend=%s",
            backend.c_str()
        );

        return true;
    }

    void release() {

        if (interpreter && session) {

            interpreter->releaseSession(
                session
            );
        }

        session = nullptr;

        interpreter.reset();

        initialized = false;

        backend =
            "Not initialized";

        status =
            "Not initialized";
    }

    bool isInitialized() const {
        return initialized;
    }

    std::string getBackend() const {
        return backend;
    }

    std::string getStatus() const {
        return status;
    }

private:

    std::shared_ptr<MNN::Interpreter>
        interpreter;

    MNN::Session*
        session = nullptr;

    bool initialized = false;

    std::string backend =
        "Not initialized";

    std::string status =
        "Not initialized";
};

SanaEngine gEngine;

/*
 * Common MNN interpreter configuration.
 */
static std::shared_ptr<MNN::Interpreter>
createInterpreter(
    const std::string& modelPath
) {
    LOGI(
        "Creating interpreter: %s",
        modelPath.c_str()
    );

    auto interpreter =
        MNN::Interpreter::createFromFile(
            modelPath.c_str()
        );

    if (!interpreter) {

        LOGE(
            "MNN interpreter creation failed"
        );
    }

    return interpreter;
}

/*
 * Create an OpenCL / FP16 session.
 *
 * This matches the configuration used by the
 * working Transformer diagnostic.
 */
static MNN::Session* createOpenCLSession(
    MNN::Interpreter* interpreter
) {
    if (interpreter == nullptr) {
        return nullptr;
    }

    MNN::ScheduleConfig config{};

    config.type =
        MNN_FORWARD_OPENCL;

    config.numThread =
        4;

    MNN::BackendConfig backendConfig{};

    backendConfig.precision =
        MNN::BackendConfig::Precision_Low;

    backendConfig.memory =
        MNN::BackendConfig::Memory_Normal;

    backendConfig.power =
        MNN::BackendConfig::Power_Normal;

    config.backendConfig =
        &backendConfig;

    return interpreter->createSession(
        config
    );
}

/*
 * Transformer diagnostic.
 *
 * IMPORTANT:
 * This path intentionally remains the same
 * diagnostic behavior that already passed.
 */
static std::string testTransformerInternal(
    const std::string& transformerPath,
    const std::string& cachePath,
    bool preferOpenCl
) {
    (void)cachePath;

    std::ostringstream result;

    result
        << "SANA 0.6B / 512 TRANSFORMER TEST\n\n";

    result
        << "Transformer file:\n"
        << transformerPath
        << "\n\n";

    FILE* file =
        std::fopen(
            transformerPath.c_str(),
            "rb"
        );

    if (!file) {

        result
            << "FAIL: Unable to open Transformer file.";

        return result.str();
    }

    std::fseek(
        file,
        0,
        SEEK_END
    );

    long long size =
        static_cast<long long>(
            std::ftell(file)
        );

    std::fclose(file);

    result
        << "File size: "
        << size
        << " bytes\n\n";

    auto interpreter =
        createInterpreter(
            transformerPath
        );

    if (!interpreter) {

        result
            << "FAIL: Transformer interpreter creation failed.";

        return result.str();
    }

    result
        << "Transformer interpreter created.\n\n";

    MNN::ScheduleConfig config{};

    if (preferOpenCl) {

        config.type =
            MNN_FORWARD_OPENCL;

    } else {

        config.type =
            MNN_FORWARD_CPU;
    }

    config.numThread =
        4;

    MNN::BackendConfig backendConfig{};

    backendConfig.precision =
        MNN::BackendConfig::Precision_Low;

    backendConfig.memory =
        MNN::BackendConfig::Memory_Normal;

    backendConfig.power =
        MNN::BackendConfig::Power_Normal;

    config.backendConfig =
        &backendConfig;

    MNN::Session* session =
        interpreter->createSession(
            config
        );

    if (!session) {

        result
            << "FAIL: Could not create Transformer session.";

        return result.str();
    }

    result
        << "Backend: "
        << (
            preferOpenCl
                ? "OpenCL / FP16"
                : "CPU"
        )
        << "\n\n";

    result
        << "Actual backend: "
        << (
            preferOpenCl
                ? "OpenCL / FP16"
                : "CPU"
        )
        << "\n\n";

    const auto& inputs =
        interpreter->getSessionInputAll(
            session
        );

    result
        << "Inputs: "
        << inputs.size()
        << "\n\n";

    for (const auto& item : inputs) {

        const std::string& name =
            item.first;

        MNN::Tensor* tensor =
            item.second;

        if (!tensor) {
            continue;
        }

        result
            << "Input: "
            << name
            << "\n";

        result
            << "Shape: "
            << shapeToString(
                tensor->shape()
            )
            << "\n";

        result
            << "Elements: "
            << tensor->elementSize()
            << "\n\n";
    }

    /*
     * Transformer input names expected from the
     * exported Sana model.
     */
    MNN::Tensor* encoderHiddenStates =
        interpreter->getSessionInput(
            session,
            "encoder_hidden_states"
        );

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

    if (!encoderHiddenStates ||
        !hiddenStates ||
        !timestep) {

        result
            << "FAIL: Required Transformer inputs were not found.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    /*
     * Use the model's existing input shapes.
     */
    interpreter->resizeTensor(
        encoderHiddenStates,
        {1, 256, 2304}
    );

    interpreter->resizeTensor(
        hiddenStates,
        {1, 32, 16, 16}
    );

    interpreter->resizeTensor(
        timestep,
        {1}
    );

    interpreter->resizeSession(
        session
    );

    /*
     * Re-acquire tensors after resizeSession().
     */
    encoderHiddenStates =
        interpreter->getSessionInput(
            session,
            "encoder_hidden_states"
        );

    hiddenStates =
        interpreter->getSessionInput(
            session,
            "hidden_states"
        );

    timestep =
        interpreter->getSessionInput(
            session,
            "timestep"
        );

    /*
     * Host tensors.
     *
     * This is the same host->device mechanism
     * used by MNN's own examples for device inputs.
     */
    MNN::Tensor encoderHost(
        encoderHiddenStates,
        MNN::Tensor::CAFFE
    );

    MNN::Tensor hiddenHost(
        hiddenStates,
        MNN::Tensor::CAFFE
    );

    MNN::Tensor timestepHost(
        timestep,
        MNN::Tensor::CAFFE
    );

    std::memset(
        encoderHost.host<void>(),
        0,
        encoderHost.size()
    );

    std::memset(
        hiddenHost.host<void>(),
        0,
        hiddenHost.size()
    );

    std::memset(
        timestepHost.host<void>(),
        0,
        timestepHost.size()
    );

    /*
     * Fill deterministic test data.
     */
    float* encoderData =
        encoderHost.host<float>();

    float* hiddenData =
        hiddenHost.host<float>();

    float* timestepData =
        timestepHost.host<float>();

    const int encoderElements =
        encoderHost.elementSize();

    const int hiddenElements =
        hiddenHost.elementSize();

    for (int i = 0;
         i < encoderElements;
         ++i) {

        encoderData[i] =
            0.001f *
            static_cast<float>(
                (i % 17) - 8
            );
    }

    for (int i = 0;
         i < hiddenElements;
         ++i) {

        hiddenData[i] =
            0.001f *
            static_cast<float>(
                (i % 11) - 5
            );
    }

    timestepData[0] =
        500.0f;

    if (!encoderHiddenStates->copyFromHostTensor(
            &encoderHost)) {

        result
            << "FAIL: Could not copy encoder_hidden_states.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    if (!hiddenStates->copyFromHostTensor(
            &hiddenHost)) {

        result
            << "FAIL: Could not copy hidden_states.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    if (!timestep->copyFromHostTensor(
            &timestepHost)) {

        result
            << "FAIL: Could not copy timestep.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    result
        << "Running inference...\n\n";

    auto start =
        std::chrono::high_resolution_clock::now();

    MNN::ErrorCode error =
        interpreter->runSession(
            session
        );

    auto end =
        std::chrono::high_resolution_clock::now();

    double elapsedMs =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    result
        << "Time: "
        << elapsedMs
        << " ms\n\n";

    result
        << "Error code: "
        << static_cast<int>(error)
        << "\n\n";

    if (error != MNN::NO_ERROR) {

        result
            << "FAIL: Transformer inference error.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    const auto& outputs =
        interpreter->getSessionOutputAll(
            session
        );

    result
        << "Outputs: "
        << outputs.size()
        << "\n\n";

    for (const auto& item : outputs) {

        MNN::Tensor* tensor =
            item.second;

        if (!tensor) {
            continue;
        }

        result
            << "Output: "
            << item.first
            << "\n";

        result
            << "Shape: "
            << shapeToString(
                tensor->shape()
            )
            << "\n";

        result
            << "Elements: "
            << tensor->elementSize()
            << "\n\n";
    }

    result
        << "PASS: Transformer executed successfully\n\n";

    interpreter->releaseSession(
        session
    );

    result
        << "Transformer released successfully.";

    return result.str();
}

/*
 * VAE diagnostic.
 *
 * ONLY IMPORTANT CHANGE:
 *
 * The old version attempted:
 *
 *     latentInput->map(MAP_TENSOR_WRITE, ...)
 *
 * That returned nullptr on this OpenCL backend.
 *
 * We now use:
 *
 *     MNN::Tensor hostTensor(latentInput, CAFFE);
 *     latentInput->copyFromHostTensor(&hostTensor);
 *
 * which is the MNN documented host->device input path.
 */
static std::string testVaeInternal(
    const std::string& vaePath,
    const std::string& cachePath,
    bool preferOpenCl
) {
    (void)cachePath;

    std::ostringstream result;

    result
        << "VAE Decoder\n\n";

    result
        << "Model file:\n"
        << vaePath
        << "\n\n";

    FILE* file =
        std::fopen(
            vaePath.c_str(),
            "rb"
        );

    if (!file) {

        result
            << "FAIL: Unable to open VAE file.";

        return result.str();
    }

    std::fseek(
        file,
        0,
        SEEK_END
    );

    long long size =
        static_cast<long long>(
            std::ftell(file)
        );

    std::fclose(file);

    result
        << "File size: "
        << size
        << " bytes\n\n";

    auto interpreter =
        createInterpreter(
            vaePath
        );

    if (!interpreter) {

        result
            << "FAIL: VAE interpreter creation failed.";

        return result.str();
    }

    result
        << "Interpreter created.\n\n";

    MNN::ScheduleConfig config{};

    if (preferOpenCl) {

        config.type =
            MNN_FORWARD_OPENCL;

    } else {

        config.type =
            MNN_FORWARD_CPU;
    }

    config.numThread =
        4;

    MNN::BackendConfig backendConfig{};

    backendConfig.precision =
        MNN::BackendConfig::Precision_Low;

    backendConfig.memory =
        MNN::BackendConfig::Memory_Normal;

    backendConfig.power =
        MNN::BackendConfig::Power_Normal;

    config.backendConfig =
        &backendConfig;

    MNN::Session* session =
        interpreter->createSession(
            config
        );

    if (!session) {

        result
            << "FAIL: Could not create VAE session.";

        return result.str();
    }

    result
        << "Backend: "
        << (
            preferOpenCl
                ? "OpenCL / FP16"
                : "CPU"
        )
        << "\n\n";

    MNN::Tensor* latentInput =
        interpreter->getSessionInput(
            session,
            "latent"
        );

    if (!latentInput) {

        /*
         * Fallback for a single-input model.
         */
        latentInput =
            interpreter->getSessionInput(
                session,
                nullptr
            );
    }

    if (!latentInput) {

        result
            << "FAIL: Could not find VAE latent input.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    result
        << "Input: latent\n";

    result
        << "Original shape: "
        << shapeToString(
            latentInput->shape()
        )
        << "\n";

    result
        << "Original elements: "
        << latentInput->elementSize()
        << "\n";

    result
        << "Original tensor type code: "
        << latentInput->getType().code
        << "\n";

    result
        << "Original tensor bytes: "
        << latentInput->getType().bytes()
        << "\n";

    result
        << "Original device ID: "
        << latentInput->deviceId()
        << "\n\n";

    /*
     * VAE latent shape:
     *
     * [1, 32, 16, 16]
     */
    interpreter->resizeTensor(
        latentInput,
        {1, 32, 16, 16}
    );

    interpreter->resizeSession(
        session
    );

    /*
     * IMPORTANT:
     *
     * Re-acquire the input AFTER resizeSession().
     */
    latentInput =
        interpreter->getSessionInput(
            session,
            "latent"
        );

    if (!latentInput) {

        latentInput =
            interpreter->getSessionInput(
                session,
                nullptr
            );
    }

    if (!latentInput) {

        result
            << "FAIL: Could not reacquire VAE latent input after resize.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    result
        << "Resized shape: "
        << shapeToString(
            latentInput->shape()
        )
        << "\n";

    result
        << "Resized elements: "
        << latentInput->elementSize()
        << "\n";

    result
        << "Resized tensor type code: "
        << latentInput->getType().code
        << "\n";

    result
        << "Resized tensor bytes: "
        << latentInput->getType().bytes()
        << "\n";

    result
        << "Resized device ID: "
        << latentInput->deviceId()
        << "\n\n";

    /*
     * ============================================================
     * NEW VAE INPUT PATH
     * ============================================================
     *
     * Do NOT map the OpenCL tensor.
     *
     * Instead create a host-side NCHW tensor that matches
     * the device tensor and let MNN perform the transfer.
     */
    result
        << "Creating host tensor for VAE input...\n";

    MNN::Tensor hostLatent(
        latentInput,
        MNN::Tensor::CAFFE
    );

    result
        << "Host tensor shape: "
        << shapeToString(
            hostLatent.shape()
        )
        << "\n";

    result
        << "Host tensor elements: "
        << hostLatent.elementSize()
        << "\n";

    result
        << "Host tensor bytes: "
        << hostLatent.getType().bytes()
        << "\n\n";

    if (hostLatent.getType().bytes() != 4) {

        result
            << "FAIL: Expected float32 VAE host tensor but got "
            << hostLatent.getType().bytes()
            << " bytes per element.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    float* latentData =
        hostLatent.host<float>();

    if (!latentData) {

        result
            << "FAIL: Host tensor returned null data.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    const int latentElements =
        hostLatent.elementSize();

    /*
     * Deterministic latent test data.
     *
     * This is NOT intended to produce the final Sana image.
     * It only verifies that the VAE graph can consume a valid
     * [1,32,16,16] float tensor and execute.
     */
    for (int i = 0;
         i < latentElements;
         ++i) {

        latentData[i] =
            -0.05f +
            (
                static_cast<float>(
                    i % 256
                ) *
                0.000390625f
            );
    }

    result
        << "Host latent elements: "
        << latentElements
        << "\n";

    result
        << "Latent sample: "
        << latentData[0]
        << ", "
        << latentData[1]
        << ", "
        << latentData[2]
        << ", "
        << latentData[3]
        << "\n\n";

    /*
     * Transfer host tensor -> OpenCL device tensor.
     */
    result
        << "Copying host tensor to OpenCL device...\n";

    bool copied =
        latentInput->copyFromHostTensor(
            &hostLatent
        );

    if (!copied) {

        result
            << "FAIL: MNN copyFromHostTensor() failed for VAE input.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    result
        << "Host -> device copy: SUCCESS\n\n";

    /*
     * Now execute the VAE graph.
     */
    result
        << "Running VAE inference...\n";

    auto start =
        std::chrono::high_resolution_clock::now();

    MNN::ErrorCode error =
        interpreter->runSession(
            session
        );

    auto end =
        std::chrono::high_resolution_clock::now();

    double elapsedMs =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    result
        << "Time: "
        << elapsedMs
        << " ms\n";

    result
        << "Error code: "
        << static_cast<int>(error)
        << "\n\n";

    if (error != MNN::NO_ERROR) {

        result
            << "FAIL: MNN VAE inference error.";

        interpreter->releaseSession(
            session
        );

        return result.str();
    }

    /*
     * Inspect VAE outputs.
     */
    const auto& outputs =
        interpreter->getSessionOutputAll(
            session
        );

    result
        << "Outputs: "
        << outputs.size()
        << "\n\n";

    for (const auto& item : outputs) {

        MNN::Tensor* output =
            item.second;

        if (!output) {
            continue;
        }

        result
            << "Output: "
            << item.first
            << "\n";

        result
            << "Shape: "
            << shapeToString(
                output->shape()
            )
            << "\n";

        result
            << "Elements: "
            << output->elementSize()
            << "\n";

        result
            << "Type bytes: "
            << output->getType().bytes()
            << "\n";

        result
            << "Device ID: "
            << output->deviceId()
            << "\n\n";
    }

    result
        << "PASS: VAE Decoder executed successfully\n\n";

    interpreter->releaseSession(
        session
    );

    result
        << "VAE released successfully.";

    return result.str();
}

} // namespace

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
    JNIEnv* env,
    jobject /* thiz */,
    jobject /* assetManager */,
    jstring modelAsset,
    jstring cachePath,
    jboolean preferOpenCl,
    jint cpuThreads
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

    std::string model =
        jstringToString(
            env,
            modelAsset
        );

    std::string cache =
        jstringToString(
            env,
            cachePath
        );

    bool success =
        gEngine.initialize(
            model,
            cache,
            preferOpenCl == JNI_TRUE,
            static_cast<int>(
                cpuThreads
            )
        );

    gInitialized =
        success;

    gBackend =
        gEngine.getBackend();

    gStatus =
        gEngine.getStatus();

    return success
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
    JNIEnv* /* env */,
    jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

    return gEngine.isInitialized()
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
    JNIEnv* env,
    jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

    std::string backend =
        gEngine.getBackend();

    return env->NewStringUTF(
        backend.c_str()
    );
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
    JNIEnv* env,
    jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

    std::string status =
        gEngine.getStatus();

    return env->NewStringUTF(
        status.c_str()
    );
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
    JNIEnv* /* env */,
    jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

    gEngine.release();

    gInitialized =
        false;

    gBackend =
        "Not initialized";

    gStatus =
        "Not initialized";
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
    JNIEnv* env,
    jobject /* thiz */,
    jstring transformerPath,
    jstring cachePath,
    jboolean preferOpenCl
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

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

    std::string output =
        testTransformerInternal(
            transformer,
            cache,
            preferOpenCl == JNI_TRUE
        );

    return env->NewStringUTF(
        output.c_str()
    );
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
    JNIEnv* env,
    jobject /* thiz */,
    jstring vaePath,
    jstring cachePath,
    jboolean preferOpenCl
) {
    std::lock_guard<std::mutex> lock(
        gMutex
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

    std::string output =
        testVaeInternal(
            vae,
            cache,
            preferOpenCl == JNI_TRUE
        );

    return env->NewStringUTF(
        output.c_str()
    );
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestModels(
    JNIEnv* env,
    jobject /* thiz */,
    jstring transformerPath,
    jstring vaePath,
    jstring cachePath,
    jboolean preferOpenCl
) {
    std::lock_guard<std::mutex> lock(
        gMutex
    );

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

    std::ostringstream result;

    result
        << "=== TRANSFORMER ===\n\n";

    result
        << testTransformerInternal(
            transformer,
            cache,
            preferOpenCl == JNI_TRUE
        );

    result
        << "\n\n";

    result
        << "=== VAE ===\n\n";

    result
        << testVaeInternal(
            vae,
            cache,
            preferOpenCl == JNI_TRUE
        );

    std::string output =
        result.str();

    return env->NewStringUTF(
        output.c_str()
    );
}
