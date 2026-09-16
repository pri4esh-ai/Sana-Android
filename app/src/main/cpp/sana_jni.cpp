#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>


// ============================================================
// Logging
// ============================================================

#define LOG_TAG "SanaNative"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


// ============================================================
// Utility
// ============================================================

static std::string shapeString(
        const std::vector<int>& shape
) {
    std::ostringstream ss;

    ss << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }

        ss << shape[i];
    }

    ss << "]";

    return ss.str();
}


// ============================================================
// Persistent Sana Engine
// ============================================================

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

        std::lock_guard<std::mutex> lock(mMutex);

        releaseLocked();


        struct stat fileInfo {};

        if (stat(
                modelPath.c_str(),
                &fileInfo
        ) != 0) {

            mStatus =
                    "Model file does not exist:\n" +
                    modelPath;

            return false;
        }


        mInterpreter.reset(
                MNN::Interpreter::createFromFile(
                        modelPath.c_str()
                )
        );


        if (!mInterpreter) {

            mStatus =
                    "Failed to create MNN interpreter.";

            return false;
        }


        MNN::ScheduleConfig config{};


        if (preferOpenCl) {

            config.type =
                    MNN_FORWARD_OPENCL;

            config.numThread =
                    cpuThreads;

            config.mode =
                    MNN_GPU_TUNING_FAST;

        } else {

            config.type =
                    MNN_FORWARD_CPU;

            config.numThread =
                    cpuThreads;

            config.mode =
                    MNN_GPU_TUNING_NONE;
        }


        mSession =
                mInterpreter->createSession(
                        config
                );


        if (!mSession) {

            mStatus =
                    "Failed to create MNN session.";

            mInterpreter.reset();

            return false;
        }


        mInitialized = true;


        if (preferOpenCl) {

            mBackend =
                    "OpenCL / FP16";

        } else {

            mBackend =
                    "CPU";
        }


        mStatus =
                "Sana engine initialized successfully.";


        return true;
    }


    bool isInitialized() {

        std::lock_guard<std::mutex> lock(mMutex);

        return mInitialized;
    }


    std::string backend() {

        std::lock_guard<std::mutex> lock(mMutex);

        return mBackend;
    }


    std::string status() {

        std::lock_guard<std::mutex> lock(mMutex);

        return mStatus;
    }


    void release() {

        std::lock_guard<std::mutex> lock(mMutex);

        releaseLocked();
    }


private:

    void releaseLocked() {

        if (mInterpreter) {

            if (mSession) {

                mInterpreter->releaseSession(
                        mSession
                );

                mSession = nullptr;
            }

            mInterpreter.reset();
        }


        mInitialized = false;

        mBackend.clear();

        mStatus =
                "Sana engine released.";
    }


private:

    std::mutex mMutex;

    std::unique_ptr<MNN::Interpreter>
            mInterpreter;

    MNN::Session* mSession =
            nullptr;

    bool mInitialized =
            false;

    std::string mBackend;

    std::string mStatus =
            "Sana engine not initialized.";
};


static SanaEngine gEngine;


// ============================================================
// Input preparation
// ============================================================

static bool prepareInput(
        MNN::Interpreter* interpreter,
        MNN::Session* session,
        MNN::Tensor* input,
        const std::string& inputName
) {

    (void) interpreter;
    (void) session;


    if (!input) {

        LOGE(
                "Input tensor is null: %s",
                inputName.c_str()
        );

        return false;
    }


    const std::vector<int> shape =
            input->shape();


    LOGI(
            "Preparing input %s shape=%s",
            inputName.c_str(),
            shapeString(shape).c_str()
    );


    MNN::Tensor hostTensor(
            input,
            MNN::Tensor::CAFFE
    );


    float* hostData =
            hostTensor.host<float>();


    if (!hostData) {

        LOGE(
                "Host tensor allocation failed: %s",
                inputName.c_str()
        );

        return false;
    }


    const size_t elementCount =
            hostTensor.elementSize();


    std::fill(
            hostData,
            hostData + elementCount,
            0.0f
    );


    // --------------------------------------------------------
    // Transformer timestep
    //
    // This is only a diagnostic value.
    // The real Sana pipeline will provide the scheduler
    // timestep during actual generation.
    // --------------------------------------------------------

    if (inputName == "timestep") {

        if (elementCount > 0) {

            hostData[0] =
                    1.0f;
        }
    }


    input->copyFromHostTensor(
            &hostTensor
    );


    return true;
}


// ============================================================
// Generic single-model diagnostic
// ============================================================

static std::string testSingleModel(
        const std::string& modelName,
        const std::string& modelPath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    (void) cachePath;


    std::ostringstream result;


    result
            << modelName
            << "\n"
            << "===\n";


    // --------------------------------------------------------
    // Check model file
    // --------------------------------------------------------

    struct stat fileInfo {};

    if (stat(
            modelPath.c_str(),
            &fileInfo
    ) != 0) {

        result
                << "FAIL: Model file does not exist.\n"
                << modelPath;

        return result.str();
    }


    result
            << "Model file:\n"
            << modelPath
            << "\n\n";


    result
            << "File size: "
            << static_cast<long long>(
                    fileInfo.st_size
            )
            << " bytes\n\n";


    // --------------------------------------------------------
    // Create interpreter
    // --------------------------------------------------------

    std::shared_ptr<MNN::Interpreter>
            interpreter(
                    MNN::Interpreter::createFromFile(
                            modelPath.c_str()
                    )
            );


    if (!interpreter) {

        result
                << "FAIL: Could not create MNN interpreter.";

        return result.str();
    }


    result
            << "Interpreter created.\n\n";


    // --------------------------------------------------------
    // Session configuration
    // --------------------------------------------------------

    MNN::ScheduleConfig config{};


    if (preferOpenCl) {

        config.type =
                MNN_FORWARD_OPENCL;

        config.numThread =
                4;

        config.mode =
                MNN_GPU_TUNING_FAST;

    } else {

        config.type =
                MNN_FORWARD_CPU;

        config.numThread =
                4;

        config.mode =
                MNN_GPU_TUNING_NONE;
    }


    MNN::BackendConfig backendConfig{};


    // --------------------------------------------------------
    // Use low precision / FP16 where supported.
    // --------------------------------------------------------

    backendConfig.precision =
            MNN::BackendConfig::Precision_Low;


    config.backendConfig =
            &backendConfig;


    MNN::Session* session =
            interpreter->createSession(
                    config
            );


    if (!session) {

        result
                << "FAIL: Could not create MNN session.";

        return result.str();
    }


    if (preferOpenCl) {

        result
                << "Backend: OpenCL / FP16\n\n";

    } else {

        result
                << "Backend: CPU\n\n";
    }


    // --------------------------------------------------------
    // Resize session
    // --------------------------------------------------------

    interpreter->resizeSession(
            session
    );


    // --------------------------------------------------------
    // Get inputs
    // --------------------------------------------------------

    std::map<std::string, MNN::Tensor*> inputs =
            interpreter->getSessionInputAll(
                    session
            );


    result
            << "Inputs: "
            << inputs.size()
            << "\n";


    if (inputs.empty()) {

        interpreter->releaseSession(
                session
        );

        return
                result.str() +
                "FAIL: No inputs found.";
    }


    // --------------------------------------------------------
    // Print inputs
    // --------------------------------------------------------

    for (const auto& pair : inputs) {

        const std::string& name =
                pair.first;

        MNN::Tensor* tensor =
                pair.second;


        result
                << "\nInput: "
                << name
                << "\n";


        if (tensor) {

            const std::vector<int> shape =
                    tensor->shape();


            result
                    << "Shape: "
                    << shapeString(shape)
                    << "\n";


            result
                    << "Elements: "
                    << tensor->elementSize()
                    << "\n";
        }
    }


    // --------------------------------------------------------
    // Prepare inputs
    // --------------------------------------------------------

    for (const auto& pair : inputs) {

        if (!prepareInput(
                interpreter.get(),
                session,
                pair.second,
                pair.first
        )) {

            interpreter->releaseSession(
                    session
            );


            return
                    result.str() +
                    "\nFAIL: Input preparation failed.";
        }
    }


    // --------------------------------------------------------
    // Run inference
    // --------------------------------------------------------

    result
            << "\nRunning inference...\n";


    const auto start =
            std::chrono::high_resolution_clock::now();


    MNN::ErrorCode errorCode =
            interpreter->runSession(
                    session
            );


    const auto end =
            std::chrono::high_resolution_clock::now();


    const double elapsedMs =
            std::chrono::duration<double, std::milli>(
                    end - start
            ).count();


    result
            << "Time: "
            << elapsedMs
            << " ms\n";


    result
            << "Error code: "
            << static_cast<int>(
                    errorCode
            )
            << "\n";


    if (errorCode != MNN::NO_ERROR) {

        interpreter->releaseSession(
                session
        );


        result
                << "FAIL: MNN inference error";

        return result.str();
    }


    // --------------------------------------------------------
    // Outputs
    // --------------------------------------------------------

    std::map<std::string, MNN::Tensor*> outputs =
            interpreter->getSessionOutputAll(
                    session
            );


    result
            << "\nOutputs: "
            << outputs.size()
            << "\n";


    for (const auto& pair : outputs) {

        const std::string& name =
                pair.first;

        MNN::Tensor* tensor =
                pair.second;


        result
                << "\nOutput: "
                << name
                << "\n";


        if (tensor) {

            const std::vector<int> shape =
                    tensor->shape();


            result
                    << "Shape: "
                    << shapeString(shape)
                    << "\n";


            result
                    << "Elements: "
                    << tensor->elementSize()
                    << "\n";
        }
    }


    // --------------------------------------------------------
    // Release session
    // --------------------------------------------------------

    interpreter->releaseSession(
            session
    );


    result
            << "\nPASS: "
            << modelName
            << " executed successfully.";


    return result.str();
}


// ============================================================
// Transformer-only diagnostic
//
// IMPORTANT:
// - Loads ONLY the Transformer.
// - Does NOT load VAE.
// ============================================================

static std::string testTransformerOnly(
        const std::string& transformerPath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    return testSingleModel(
            "Transformer",
            transformerPath,
            cachePath,
            preferOpenCl
    );
}


// ============================================================
// Combined Transformer + VAE diagnostic
// ============================================================

static std::string testModels(
        const std::string& transformerPath,
        const std::string& vaePath,
        const std::string& cachePath,
        bool preferOpenCl
) {

    std::ostringstream result;


    result
            << "SANA MODEL TEST\n"
            << "================\n\n";


    // --------------------------------------------------------
    // Transformer
    // --------------------------------------------------------

    result
            << "TRANSFORMER TEST\n"
            << "----------------\n";


    result
            << testSingleModel(
                    "Transformer",
                    transformerPath,
                    cachePath,
                    preferOpenCl
            );


    result
            << "\n\n";


    // --------------------------------------------------------
    // VAE
    // --------------------------------------------------------

    result
            << "VAE TEST\n"
            << "--------\n";


    result
            << testSingleModel(
                    "VAE Decoder",
                    vaePath,
                    cachePath,
                    preferOpenCl
            );


    return result.str();
}


// ============================================================
// JNI: nativeInitialize
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
        jint cpuThreads
) {

    (void) assetManager;


    try {

        if (modelAsset == nullptr) {

            return JNI_FALSE;
        }


        const char* modelChars =
                env->GetStringUTFChars(
                        modelAsset,
                        nullptr
                );


        if (!modelChars) {

            return JNI_FALSE;
        }


        std::string model =
                modelChars;


        env->ReleaseStringUTFChars(
                modelAsset,
                modelChars
        );


        std::string cache;


        if (cachePath != nullptr) {

            const char* cacheChars =
                    env->GetStringUTFChars(
                            cachePath,
                            nullptr
                    );


            if (cacheChars) {

                cache =
                        cacheChars;


                env->ReleaseStringUTFChars(
                        cachePath,
                        cacheChars
                );
            }
        }


        return gEngine.initialize(
                model,
                cache,
                preferOpenCl == JNI_TRUE,
                static_cast<int>(
                        cpuThreads
                )
        )
                ? JNI_TRUE
                : JNI_FALSE;

    } catch (...) {

        return JNI_FALSE;
    }
}


// ============================================================
// JNI: nativeIsInitialized
// ============================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sana_android_engine_NativeSana_nativeIsInitialized(
        JNIEnv*,
        jobject
) {

    return gEngine.isInitialized()
            ? JNI_TRUE
            : JNI_FALSE;
}


// ============================================================
// JNI: nativeGetBackend
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetBackend(
        JNIEnv* env,
        jobject
) {

    const std::string backend =
            gEngine.backend();


    return env->NewStringUTF(
            backend.c_str()
    );
}


// ============================================================
// JNI: nativeGetStatus
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeGetStatus(
        JNIEnv* env,
        jobject
) {

    const std::string status =
            gEngine.status();


    return env->NewStringUTF(
            status.c_str()
    );
}


// ============================================================
// JNI: nativeRelease
// ============================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_sana_android_engine_NativeSana_nativeRelease(
        JNIEnv*,
        jobject
) {

    gEngine.release();
}


// ============================================================
// JNI: nativeTestTransformer
//
// Transformer-only diagnostic.
//
// IMPORTANT:
// - Loads ONLY sana_transformer.mnn.
// - Does NOT load VAE.
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestTransformer(
        JNIEnv* env,
        jobject,
        jstring transformerPath,
        jstring cachePath,
        jboolean preferOpenCl
) {

    try {

        if (transformerPath == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Transformer path is null."
            );
        }


        const char* transformerChars =
                env->GetStringUTFChars(
                        transformerPath,
                        nullptr
                );


        if (transformerChars == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Unable to read Transformer path."
            );
        }


        std::string transformer =
                transformerChars;


        env->ReleaseStringUTFChars(
                transformerPath,
                transformerChars
        );


        std::string cache;


        if (cachePath != nullptr) {

            const char* cacheChars =
                    env->GetStringUTFChars(
                            cachePath,
                            nullptr
                    );


            if (cacheChars != nullptr) {

                cache =
                        cacheChars;


                env->ReleaseStringUTFChars(
                        cachePath,
                        cacheChars
                );
            }
        }


        const std::string output =
                testTransformerOnly(
                        transformer,
                        cache,
                        preferOpenCl == JNI_TRUE
                );


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (const std::exception& e) {

        std::string message =
                "C++ exception: ";

        message +=
                e.what();


        return env->NewStringUTF(
                message.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown native C++ exception."
        );
    }
}


// ============================================================
// JNI: nativeTestVae
//
// VAE-only diagnostic.
//
// IMPORTANT:
// - Loads ONLY sana_vae_decoder.mnn.
// - Does NOT load Transformer.
// - Does NOT create a Transformer session.
// ============================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sana_android_engine_NativeSana_nativeTestVae(
        JNIEnv* env,
        jobject,
        jstring vaePath,
        jstring cachePath,
        jboolean preferOpenCl
) {

    try {

        if (vaePath == nullptr) {

            return env->NewStringUTF(
                    "FAIL: VAE path is null."
            );
        }


        const char* vaeChars =
                env->GetStringUTFChars(
                        vaePath,
                        nullptr
                );


        if (vaeChars == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Unable to read VAE path."
            );
        }


        std::string vae =
                vaeChars;


        env->ReleaseStringUTFChars(
                vaePath,
                vaeChars
        );


        std::string cache;


        if (cachePath != nullptr) {

            const char* cacheChars =
                    env->GetStringUTFChars(
                            cachePath,
                            nullptr
                    );


            if (cacheChars != nullptr) {

                cache =
                        cacheChars;


                env->ReleaseStringUTFChars(
                        cachePath,
                        cacheChars
                );
            }
        }


        // ----------------------------------------------------
        // Test ONLY the VAE.
        // ----------------------------------------------------

        const std::string output =
                testSingleModel(
                        "VAE Decoder",
                        vae,
                        cache,
                        preferOpenCl == JNI_TRUE
                );


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (const std::exception& e) {

        std::string message =
                "C++ exception: ";

        message +=
                e.what();


        return env->NewStringUTF(
                message.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown native C++ exception."
        );
    }
}


// ============================================================
// JNI: nativeTestModels
//
// Legacy combined diagnostic.
// Kept for compatibility with existing Kotlin code.
// ============================================================

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

    try {

        if (transformerPath == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Transformer path is null."
            );
        }


        if (vaePath == nullptr) {

            return env->NewStringUTF(
                    "FAIL: VAE path is null."
            );
        }


        const char* transformerChars =
                env->GetStringUTFChars(
                        transformerPath,
                        nullptr
                );


        if (transformerChars == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Unable to read Transformer path."
            );
        }


        std::string transformer =
                transformerChars;


        env->ReleaseStringUTFChars(
                transformerPath,
                transformerChars
        );


        const char* vaeChars =
                env->GetStringUTFChars(
                        vaePath,
                        nullptr
                );


        if (vaeChars == nullptr) {

            return env->NewStringUTF(
                    "FAIL: Unable to read VAE path."
            );
        }


        std::string vae =
                vaeChars;


        env->ReleaseStringUTFChars(
                vaePath,
                vaeChars
        );


        std::string cache;


        if (cachePath != nullptr) {

            const char* cacheChars =
                    env->GetStringUTFChars(
                            cachePath,
                            nullptr
                    );


            if (cacheChars != nullptr) {

                cache =
                        cacheChars;


                env->ReleaseStringUTFChars(
                        cachePath,
                        cacheChars
                );
            }
        }


        const std::string output =
                testModels(
                        transformer,
                        vae,
                        cache,
                        preferOpenCl == JNI_TRUE
                );


        return env->NewStringUTF(
                output.c_str()
        );

    } catch (const std::exception& e) {

        std::string message =
                "C++ exception: ";

        message +=
                e.what();


        return env->NewStringUTF(
                message.c_str()
        );

    } catch (...) {

        return env->NewStringUTF(
                "Unknown native C++ exception."
        );
    }
}
