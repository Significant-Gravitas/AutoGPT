# Native chat notifications

The apps open `/mobile`, with **Chats**, **Experts**, and **Needs you**. Selecting a conversation or expert opens the existing hosted chat. Questions, editable reviews, approvals, setup requests, and paused work use the same controls and authorization as the website.

Native push is optional and ships unconfigured. The app remains usable without it. This PR creates no Apple Developer team, push key, Firebase project, release signing identity, or store release.

## Server prerequisites

Deploy both the frontend and backend from this change and apply Prisma migrations, including `20261007220000_native_push`. The frontend's Better Auth database connection must have access to `NativePushSubscription` in the same schema as `UserAuthSession`.

Mount provider-issued credentials as read-only files outside the repository. Configure these optional backend settings on services that publish notifications, including chat workers and the database manager:

| Setting                    | Value                                                      |
| -------------------------- | ---------------------------------------------------------- |
| `APNS_PRIVATE_KEY_PATH`    | Absolute path to the mounted Apple `.p8` key               |
| `APNS_KEY_ID`              | The Apple push key's ID                                    |
| `APNS_TEAM_ID`             | Your Apple Developer team ID                               |
| `FCM_SERVICE_ACCOUNT_PATH` | Absolute path to the mounted Firebase service-account JSON |

These settings contain file paths and public identifiers, not private keys. Provider credentials must be issued by Apple or Google; `make init-env` does not create them. Leave paths empty to disable each transport. Never put key files, service-account JSON, or working private key values in `.env.default`, build resources, screenshots, or git.

`GET /api/push/native/config` reports whether settings are present. This is a configuration check, not a delivery test. Allow outbound HTTPS to Apple's APNs endpoints and Google's OAuth/FCM endpoints; APNs uses HTTP/2.

## Apple setup

1. Enroll in the Apple Developer Program and choose the signing team in Xcode. Register app identifier `com.agpt.mobile` with the Push Notifications capability.
2. Create an APNs authentication key in the developer account. Store the downloaded `.p8` file in your secret manager. Mount it on the backend and set the three APNs settings above.
3. Regenerate the Xcode project and build with the chosen `DEVELOPMENT_TEAM`. The project includes the `aps-environment` entitlement: Debug uses `development` and the sandbox APNs endpoint; Release uses `production`. The provisioning profile must include the same capability/environment. Do not change just one setting.
4. Install the signed build, sign in to the matching server, open **Chats, experts & requests**, and choose **Enable notifications**. Grant permission when iOS asks.

Unsigned simulator compilation does not prove APNs registration. Use a signed device build for delivery validation. If you change the bundle identifier, update the server's APNs topic and native identifiers together.

Apple references: [register with APNs](https://developer.apple.com/documentation/usernotifications/registering-your-app-with-apns), [send notification requests](https://developer.apple.com/documentation/usernotifications/sending-notification-requests-to-apns).

## Android setup

1. Create a Firebase project and register an Android app with package `com.agpt.mobile`. Enable the Firebase Cloud Messaging HTTP v1 API.
2. Supply the public Android configuration through local Gradle properties or CI settings:

   | Gradle property               | Firebase configuration field |
   | ----------------------------- | ---------------------------- |
   | `AUTOGPT_FIREBASE_APP_ID`     | `mobilesdk_app_id`           |
   | `AUTOGPT_FIREBASE_API_KEY`    | `api_key[].current_key`      |
   | `AUTOGPT_FIREBASE_PROJECT_ID` | `project_id`                 |
   | `AUTOGPT_FIREBASE_SENDER_ID`  | `project_number`             |

3. Create a backend service account with permission to send messages for the project. Store its JSON in your secret manager and mount it at `FCM_SERVICE_ACCOUNT_PATH`. This private file is separate from the public Android configuration and must never enter the APK.
4. Build and install on a device with Google Play services or a Google APIs emulator. Sign in, choose **Enable notifications**, and grant notification permission on Android 13+.

Firebase initialization is skipped when any build property is missing. Automatic token collection is disabled; opting in obtains a token. Token refresh updates the server using the current app session, and opening the app retries registration. Android receives data messages and creates the notification after checking the active server and account binding.

Google references: [Android setup](https://firebase.google.com/docs/cloud-messaging/android/get-started), [receive messages](https://firebase.google.com/docs/cloud-messaging/android/receive-messages), [HTTP v1 authentication](https://firebase.google.com/docs/cloud-messaging/send/v1-api).

## Behavior and validation

- Chat completion/failure produces a generic update. A persisted question or pending human review produces a request for attention. Review pushes are deduplicated for 24 hours. **Needs you** reloads current requests; notifications are hints, not the source of truth.
- Payloads contain a generic message, relative route, configured origin, and opaque binding ID. They contain no conversation text, question text, approval payload, credentials, or session cookie. Tapping never approves an action automatically.
- Registration requires a live, authoritative Better Auth session and matching account ID. The database foreign key deletes registrations when their session is revoked or signed out. Expired sessions, impersonated sessions, and banned users are excluded from delivery.
- Signing in again or changing servers clears the local binding and attempts to remove the old registration. Offline removal can leave a generic iOS alert until the old session is revoked or expires; its tap cannot enter the new account/server. Android suppresses mismatched bindings before display.
- Delivery is best effort through the existing notification event bus. Transient failures retain registrations for the next event; permanently invalid tokens are removed. This is not a durable retry queue. Provider messages expire after one hour.

Before enabling this in a release, verify on signed iOS and configured Android builds:

1. Sign in and enable notifications. Check permission-denied and retry states.
2. Start an expert conversation. Background the app, cause a question or approval, and tap its notification. Confirm the correct conversation/request opens and can be answered, edited, approved, or declined.
3. Repeat with the app terminated and with a normal chat completion. Confirm requests answered on the website no longer appear as pending.
4. Sign out, change accounts, and switch servers. Confirm old notification taps are ignored and only the current session receives new notifications.
5. Revoke notification permission, re-enable it in system settings, and reopen the app. Confirm registration and token refresh recover.

Local fixtures and policy tests do not demonstrate provider delivery. No provider credentials are configured in this checkout; live APNs/FCM delivery remains unverified.
