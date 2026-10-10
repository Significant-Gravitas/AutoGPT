package com.agpt.mobile

import android.app.Application
import com.google.firebase.FirebaseApp
import com.google.firebase.FirebaseOptions

class AutoGPTApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        if (
            listOf(
                    BuildConfig.FIREBASE_APP_ID,
                    BuildConfig.FIREBASE_API_KEY,
                    BuildConfig.FIREBASE_PROJECT_ID,
                    BuildConfig.FIREBASE_SENDER_ID,
                )
                .any { it.isBlank() }
        )
            return
        FirebaseApp.initializeApp(
            this,
            FirebaseOptions.Builder()
                .setApplicationId(BuildConfig.FIREBASE_APP_ID)
                .setApiKey(BuildConfig.FIREBASE_API_KEY)
                .setProjectId(BuildConfig.FIREBASE_PROJECT_ID)
                .setGcmSenderId(BuildConfig.FIREBASE_SENDER_ID)
                .build(),
        )
    }
}
