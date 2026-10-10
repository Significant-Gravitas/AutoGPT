package com.agpt.mobile

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class PushService : FirebaseMessagingService() {
    override fun onNewToken(token: String) {
        PushRegistration.sync(this, token)
    }

    override fun onMessageReceived(message: RemoteMessage) {
        val prefs = getSharedPreferences("push", MODE_PRIVATE)
        if (
            !prefs.getBoolean("enabled", false) ||
                !NotificationManagerCompat.from(this).areNotificationsEnabled()
        )
            return
        val origin =
            ServerOrigin.parse(prefs.getString("origin", "") ?: "", BuildConfig.DEBUG) ?: return
        val data = message.data
        val path = data["path"] ?: return
        val binding = data["binding_id"] ?: return
        PushTarget.url(
            path,
            data["origin"] ?: "",
            binding,
            prefs.getString("binding", "") ?: "",
            origin,
        ) ?: return
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                "chat_updates",
                "Chats and requests",
                NotificationManager.IMPORTANCE_DEFAULT,
            )
        )
        val notificationID = (message.messageId ?: path).hashCode()
        val intent =
            Intent(this, MainActivity::class.java)
                .setAction(ACTION)
                .putExtra("push_path", path)
                .putExtra("push_origin", origin.value)
                .putExtra("push_binding", binding)
        val pending =
            PendingIntent.getActivity(
                this,
                notificationID,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
        val notification =
            NotificationCompat.Builder(this, "chat_updates")
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle("AutoGPT")
                .setContentText("Your team has an update. Open AutoGPT to continue.")
                .setContentIntent(pending)
                .setAutoCancel(true)
                .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
                .build()
        try {
            manager.notify(notificationID, notification)
        } catch (_: SecurityException) {}
    }

    companion object {
        const val ACTION = "com.agpt.mobile.PUSH"
    }
}
