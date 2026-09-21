plugins {
    id("com.android.application") version "9.4.0"
    id("org.jetbrains.kotlin.plugin.serialization") version "2.4.10"
}

android {
    namespace = "com.avtracker.mobile"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.avtracker.mobile"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        // Phones (arm64) and the x86_64 emulator; dropping the 32-bit ONNX Runtime libraries saves ~25 MB.
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.all { it.maxHeapSize = "2g" }
        // android.util.Log calls in the engine become no-ops in JVM unit tests instead of throwing.
        unitTests.isReturnDefaultValues = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    testImplementation("junit:junit:4.13.2")
    // Desktop ONNX Runtime so JVM unit tests can execute the exported models for real.
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
    // the JVM tests run TrackerDb's SQL on SQLite itself
    testImplementation("org.xerial:sqlite-jdbc:3.46.1.3")
    testImplementation(kotlin("test-junit"))
}
