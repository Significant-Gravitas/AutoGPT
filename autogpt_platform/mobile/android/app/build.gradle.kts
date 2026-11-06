import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val appOrigin = providers.gradleProperty("autogptOrigin").getOrElse("https://platform.agpt.co")
val originURI = URI(appOrigin)
require(originURI.scheme == "https" && originURI.host != null && originURI.rawUserInfo == null
    && originURI.rawQuery == null && originURI.rawFragment == null && originURI.rawPath.isNullOrEmpty()) {
    "autogptOrigin must be an HTTPS origin without a path or trailing slash"
}

android {
    namespace = "co.agpt.autogpt"
    compileSdk = 36
    defaultConfig {
        applicationId = "co.agpt.autogpt"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("String", "APP_ORIGIN", "\"$appOrigin\"")
        manifestPlaceholders["appHost"] = originURI.host
    }
    buildFeatures { buildConfig = true }
    buildTypes {
        debug { applicationIdSuffix = ".dev" }
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.browser:browser:1.8.0")
    implementation("androidx.core:core-ktx:1.15.0")
    testImplementation("junit:junit:4.13.2")
}
