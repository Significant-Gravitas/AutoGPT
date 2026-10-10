plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.agpt.mobile"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.agpt.mobile"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "com.agpt.mobile.RuntimeProbe"
        listOf("APP_ID", "API_KEY", "PROJECT_ID", "SENDER_ID").forEach { key ->
            val value = providers.gradleProperty("AUTOGPT_FIREBASE_$key").orElse("").get()
            require(value.none { it == '"' || it == '\\' || it.isISOControl() })
            buildConfigField("String", "FIREBASE_$key", "\"$value\"")
        }
    }

    buildFeatures {
        buildConfig = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    lint {
        abortOnError = true
        checkReleaseBuilds = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("com.google.firebase:firebase-messaging:25.1.3")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.browser:browser:1.9.0")
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.fragment:fragment-ktx:1.8.9")
    implementation("androidx.webkit:webkit:1.14.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.9.1")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
}
