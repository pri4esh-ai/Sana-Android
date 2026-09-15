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

      # --------------------------------------------------
      # 1. Checkout repository
      # --------------------------------------------------
      - name: Checkout repository
        uses: actions/checkout@v4

      # --------------------------------------------------
      # 2. Java 17
      # --------------------------------------------------
      - name: Setup Java 17
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '17'

      # --------------------------------------------------
      # 3. Install Android SDK
      # --------------------------------------------------
      - name: Install Android SDK
        shell: bash
        run: |
          set -e

          SDK_ROOT="$HOME/android-sdk"

          mkdir -p "$SDK_ROOT"

          cd "$HOME"

          wget -q \
            https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip \
            -O commandlinetools.zip

          mkdir -p "$SDK_ROOT/cmdline-tools"

          unzip -q \
            commandlinetools.zip \
            -d "$SDK_ROOT/cmdline-tools"

          mv \
            "$SDK_ROOT/cmdline-tools/cmdline-tools" \
            "$SDK_ROOT/cmdline-tools/latest"

          echo "ANDROID_HOME=$SDK_ROOT" >> "$GITHUB_ENV"
          echo "ANDROID_SDK_ROOT=$SDK_ROOT" >> "$GITHUB_ENV"

          echo "$SDK_ROOT/cmdline-tools/latest/bin" >> "$GITHUB_PATH"
          echo "$SDK_ROOT/platform-tools" >> "$GITHUB_PATH"

      # --------------------------------------------------
      # 4. Verify SDK
      # --------------------------------------------------
      - name: Verify sdkmanager
        shell: bash
        run: |
          set -e

          echo "ANDROID_HOME=$ANDROID_HOME"

          which sdkmanager

          sdkmanager --version

      # --------------------------------------------------
      # 5. Accept licenses
      # --------------------------------------------------
      - name: Accept Android licenses
        shell: bash
        run: |
          yes | sdkmanager --licenses > /dev/null || true

      # --------------------------------------------------
      # 6. Install Android packages
      # --------------------------------------------------
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

      # --------------------------------------------------
      # 7. Verify native toolchain
      # --------------------------------------------------
      - name: Verify native toolchain
        shell: bash
        run: |
          set -e

          NDK="$ANDROID_HOME/ndk/27.2.12479018"
          CMAKE="$ANDROID_HOME/cmake/3.31.6/bin/cmake"

          echo "========================================"
          echo "Checking NDK"
          echo "========================================"

          test -d "$NDK"
          echo "NDK found:"
          echo "$NDK"

          echo ""
          echo "NDK version:"
          "$NDK/ndk-build" --version || true

          echo ""
          echo "========================================"
          echo "Checking Android CMake toolchain"
          echo "========================================"

          test -f \
            "$NDK/build/cmake/android.toolchain.cmake"

          echo "android.toolchain.cmake found"

          echo ""
          echo "========================================"
          echo "Checking CMake"
          echo "========================================"

          "$CMAKE" --version

      # --------------------------------------------------
      # 8. Check native source files
      # --------------------------------------------------
      - name: Verify Sana native files
        shell: bash
        run: |
          set -e

          echo "Checking CMakeLists.txt..."

          test -f \
            app/src/main/cpp/CMakeLists.txt

          echo "Found CMakeLists.txt"

          echo ""
          echo "Checking sana_jni.cpp..."

          test -f \
            app/src/main/cpp/sana_jni.cpp

          echo "Found sana_jni.cpp"

          echo ""
          echo "Native source tree:"
          find app/src/main/cpp -maxdepth 2 -type f -print

      # --------------------------------------------------
      # 9. Setup Gradle
      # --------------------------------------------------
      - name: Setup Gradle
        uses: gradle/actions/setup-gradle@v4
        with:
          gradle-version: '8.9'

      # --------------------------------------------------
      # 10. Gradle version
      # --------------------------------------------------
      - name: Gradle version
        shell: bash
        run: |
          java -version
          gradle --version

      # --------------------------------------------------
      # 11. Build APK
      # --------------------------------------------------
      - name: Build Debug APK
        shell: bash
        run: |
          set -e

          gradle \
            assembleDebug \
            --no-daemon \
            --stacktrace

      # --------------------------------------------------
      # 12. Verify APK
      # --------------------------------------------------
      - name: Verify APK
        shell: bash
        run: |
          set -e

          APK="app/build/outputs/apk/debug/app-debug.apk"

          test -f "$APK"

          echo "========================================"
          echo "APK BUILD SUCCESS"
          echo "========================================"

          ls -lh "$APK"

      # --------------------------------------------------
      # 13. Upload APK
      # --------------------------------------------------
      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: Sana-Android-debug
          path: app/build/outputs/apk/debug/app-debug.apk
          if-no-files-found: error
