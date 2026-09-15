name: Build Sana Android

on:
  workflow_dispatch:
  push:
    branches:
      - main

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-22.04

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Java 17
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '17'

      - name: Install Android SDK
        shell: bash
        run: |
          set -e

          SDK_ROOT="$HOME/android-sdk"
          mkdir -p "$SDK_ROOT"

          cd "$HOME"

          wget -q https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip -O commandlinetools.zip

          mkdir -p "$SDK_ROOT/cmdline-tools"

          unzip -q commandlinetools.zip -d "$SDK_ROOT/cmdline-tools"

          mv "$SDK_ROOT/cmdline-tools/cmdline-tools" "$SDK_ROOT/cmdline-tools/latest"

          echo "ANDROID_HOME=$SDK_ROOT" >> "$GITHUB_ENV"
          echo "ANDROID_SDK_ROOT=$SDK_ROOT" >> "$GITHUB_ENV"

          echo "$SDK_ROOT/cmdline-tools/latest/bin" >> "$GITHUB_PATH"
          echo "$SDK_ROOT/platform-tools" >> "$GITHUB_PATH"

      - name: Verify sdkmanager
        shell: bash
        run: |
          which sdkmanager
          sdkmanager --version

      - name: Accept Android licenses
        shell: bash
        run: |
          yes | sdkmanager --licenses > /dev/null || true

      - name: Install Android packages
        shell: bash
        run: |
          set -e

          sdkmanager \
            "platform-tools" \
            "platforms;android-35" \
            "build-tools;35.0.0" \
            "cmake;3.31.6" \
            "ndk;27.2.12479018"

      - name: Clone MNN
        shell: bash
        run: |
          git clone --depth 1 https://github.com/alibaba/MNN.git mnn

      - name: Build MNN
        shell: bash
        env:
          ANDROID_NDK: ${{ env.ANDROID_HOME }}/ndk/27.2.12479018
        run: |
          set -e

          mkdir -p mnn/build-android
          cd mnn/build-android

          cmake .. \
            -G Ninja \
            -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_PLATFORM=android-26 \
            -DCMAKE_BUILD_TYPE=Release \
            -DMNN_OPENCL=ON \
            -DMNN_ARM82=ON \
            -DMNN_LOW_MEMORY=ON \
            -DMNN_BUILD_LLM=ON \
            -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

          cmake --build . --parallel 2

      - name: Verify MNN
        shell: bash
        run: |
          find mnn/build-android \( -name "libMNN*.so" -o -name "libMNN_Express*.so" \)

      - name: Setup Gradle
        uses: gradle/actions/setup-gradle@v4
        with:
          gradle-version: '8.9'

      - name: Gradle version
        run: |
          java -version
          gradle --version

      - name: Build Debug APK
        shell: bash
        run: |
          gradle assembleDebug --no-daemon --stacktrace

      - name: Verify APK
        shell: bash
        run: |
          test -f app/build/outputs/apk/debug/app-debug.apk
          ls -lh app/build/outputs/apk/debug/app-debug.apk

      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: Sana-Android-debug
          path: app/build/outputs/apk/debug/app-debug.apk
          if-no-files-found: error
