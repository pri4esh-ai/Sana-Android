#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "SanaNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

class SanaEngine {
public:
    SanaEngine() = default;
    ~SanaEngine() { release(); }

    bool initialize(
            const std::string& modelPath,
            const std::string& cachePath,
            bool preferOpenCl,
            int cpuThreads
    ) {

        std::lock_guard<std::mutex> lock(mMutex);

        if (mInitialized) return true;

        if (modelPath.empty()) {
            mLastError = "Empty model path";
            return false;
        }

        mInterpreter.reset(
                MNN::Interpreter::createFromFile(modelPath.c_str())
        );

        if (!mInterpreter) {
            mLastError = "Failed to load model";
            return false;
        }

        if (!cachePath.empty()) {
            mCacheFile = cachePath + "/sana_gpu.cache";
            mInterpreter->setCacheFile(mCacheFile.c_str());
        }

        MNN::ScheduleConfig config;
        MNN::BackendConfig backend;

        backend.precision = MNN::BackendConfig::Precision_Low;
        backend.memory = MNN::BackendConfig::Memory_Low;
        config.backendConfig = &backend;

        if (preferOpenCl) {
            config.type = MNN_FORWARD_OPENCL;
            config.numThread = 1;
            mSession = mInterpreter->createSession(config);
        }

        if (!mSession) {
            config.type = MNN_FORWARD_CPU;
            config.numThread = std::max(1, std::min(cpuThreads, 8));
            mSession = mInterpreter->createSession(config);
            mBackend = "CPU";
        } else {
            mBackend = "OpenCL FP16";
        }

        if (!mSession) {
            mInterpreter.reset();
            mLastError = "Session creation failed";
            return false;
        }

        mInitialized = true;
        return true;
    }

    void release() {

        std::lock_guard<std::mutex> lock(mMutex);

        if (mInterpreter && mSession)
            mInterpreter->releaseSession(mSession);

        mSession = nullptr;
        mInterpreter.reset();
        mInitialized = false;
    }

    bool initialized() const {
        return mInitialized;
    }

    std::string backend() const {
        return mBackend;
    }

    std::string status() const {

        if (mInitialized)
            return "Ready | " + mBackend;

        if (!mLastError.empty())
            return "Error | " + mLastError;

        return "Not initialized";
    }

private:
    mutable std::mutex mMutex;

    std::unique_ptr<MNN::Interpreter> mInterpreter;
    MNN::Session* mSession = nullptr;

    std::string mBackend = "None";
    std::string mLastError;
    std::string mCacheFile;
    bool mInitialized = false;
};

SanaEngine gEngine;

std::string shapeToString(const std::vector<int>& shape) {

    std::ostringstream out;
    out << "[";

    for (size_t i = 0; i < shape.size(); i++) {

        out << shape[i];

        if (i + 1 < shape.size())
            out << ", ";
    }

    out << "]";
    return out.str();
}

/*
 * Fill tensor AFTER resize.
 */
void fillInput(
        MNN::Tensor* deviceTensor,
        const std::string& name
) {

    MNN::Tensor host(deviceTensor, MNN::Tensor::CAFFE);

    float* ptr = host.host<float>();

    if (!ptr)
        return;

    const int elements = host.elementSize();

    std::fill(ptr, ptr + elements, 0.f);

    if (name == "timestep" && elements > 0)
        ptr[0] = 1.f;

    deviceTensor->copyFromHostTensor(&host);
}

std::string runModel(
        const std::string& name,
        const std::string& path,
        const std::string& cachePath,
        bool preferOpenCl
) {

    std::ostringstream out;

    out << "========================================\n";
    out << name << "\n";
    out << "========================================\n";

    auto interpreter = std::unique_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(path.c_str())
    );

    if (!interpreter) {
        out << "FAIL: load failed\n";
        return out.str();
    }

    if (!cachePath.empty()) {

        interpreter->setCacheFile(
                (cachePath + "/" + name + ".cache").c_str()
        );
    }

    MNN::ScheduleConfig config;
    MNN::BackendConfig backend;

    backend.precision = MNN::BackendConfig::Precision_Low;
    backend.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backend;

    config.type = preferOpenCl
                  ? MNN_FORWARD_OPENCL
                  : MNN_FORWARD_CPU;

    config.numThread = preferOpenCl ? 1 : 4;

    MNN::Session* session = interpreter->createSession(config);

    std::string backendName;

    if (session) {

        backendName = preferOpenCl
                      ? "OpenCL / FP16"
                      : "CPU";

    } else {

        config.type = MNN_FORWARD_CPU;
        config.numThread = 4;

        session = interpreter->createSession(config);

        backendName = "CPU";
    }

    if (!session) {

        out << "FAIL: session failed\n";
        return out.str();
    }

    out << "Backend: " << backendName << "\n";

    auto inputs = interpreter->getSessionInputAll(session);

    out << "Inputs: " << inputs.size() << "\n\n";

    /*
     * IMPORTANT FIX
     *
     * Resize first.
     */
    interpreter->resizeSession(session);

    for (auto& kv : inputs) {

        auto* tensor = kv.second;

        out << "Input: " << kv.first << "\n";
        out << "Shape: " << shapeToString(tensor->shape()) << "\n";
        out << "Elements: " << tensor->elementSize() << "\n";

        fillInput(tensor, kv.first);
    }

    auto t0 = std::chrono::steady_clock::now();

    auto code = interpreter->runSession(session);

    auto t1 = std::chrono::steady_clock::now();

    double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

    out << "\nRunning inference...\n";
    out << "Time: " << ms << " ms\n";
    out << "Error code: " << (int)code << "\n";

    if (code != MNN::NO_ERROR) {

        out << "FAIL: inference error\n";
        interpreter->releaseSession(session);
        return out.str();
    }

    auto outputs = interpreter->getSessionOutputAll(session);

    out << "Outputs: " << outputs.size() << "\n";

    for (auto& kv : outputs) {

        auto* tensor = kv.second;

        out << "Output: " << kv.first << "\n";
        out << "Shape: " << shapeToString(tensor->shape()) << "\n";
        out << "Elements: " << tensor->elementSize() << "\n";
    }

    out << "\nPASS: model executed\n";

    interpreter->releaseSession(session);

    return out.str();
}

std::string testModels(
        const std::string& transformer,
        const std::string& vae,
        const std::string& cache,
        bool opencl
) {

    std::ostringstream out;

    out << "SANA 0.6B / 512 MODEL TEST\n\n";

    out << "Transformer file:\n";
    out << transformer << "\n\n";

    out << "VAE file:\n";
    out << vae << "\n\n";

    out << runModel(
            "transformer",
            transformer,
            cache,
            opencl
    );

    out << "\n";

    out << runModel(
            "vae_decoder",
            vae,
            cache,
            opencl
    );

    out << "\n========================================\n";
    out << "TEST COMPLETE\n";
    out << "========================================\n";

    return out.str();
}

} // namespace

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeInitialize(
        JNIEnv*,
        jobject,
        jobject,
        jstring,
        jstring,
        jboolean,
        jint
) {
    return JNI_FALSE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv*,
        jobject
) {
    return gEngine.initialized();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject
) {
    return env->NewStringUTF(gEngine.backend().c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject
) {
    return env->NewStringUTF(gEngine.status().c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject
) {
    gEngine.release();
}

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

    const char* t = env->GetStringUTFChars(transformerPath, nullptr);
    const char* v = env->GetStringUTFChars(vaePath, nullptr);

    const char* c =
            cachePath
            ? env->GetStringUTFChars(cachePath, nullptr)
            : "";

    std::string output = testModels(
            t,
            v,
            c,
            preferOpenCl == JNI_TRUE
    );

    env->ReleaseStringUTFChars(transformerPath, t);
    env->ReleaseStringUTFChars(vaePath, v);

    if (cachePath)
        env->ReleaseStringUTFChars(cachePath, c);

    return env->NewStringUTF(output.c_str());
}
