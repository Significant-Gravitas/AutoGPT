import AuthenticationServices
import StoreKit
import UIKit
import UserNotifications

/// Whether a provider integration can be offered. An unavailable provider
/// gets an explanation, never a button that can't complete.
enum ProviderReadiness: Equatable, Sendable {
  case available
  case unavailable(String)

  var explanation: String? {
    if case .unavailable(let reason) = self { return reason }
    return nil
  }
}

/// Google sign-in will run in the system authentication browser with fresh
/// state and PKCE, completing through the server's bound one-use native
/// handoff. That server contract isn't part of PR 1 yet, so no server
/// advertises it and the app offers no Google button.
enum GoogleSignIn {
  static func readiness(for server: ServerConfig) -> ProviderReadiness {
    .unavailable(
      "Signing in with Google isn't available in the app yet. If your account only uses Google, sign in on the web for now."
    )
  }
}

/// The system authentication browser for provider handoffs (Google sign-in,
/// connection OAuth). Each session is ephemeral, so no browser cookie links
/// it to another account. Not reachable from the UI until a server
/// advertises a native handoff.
@MainActor
final class SystemBrowserSession: NSObject {
  private var session: ASWebAuthenticationSession?

  func start(url: URL, callbackScheme: String) async throws -> URL {
    try await withCheckedThrowingContinuation { continuation in
      let session = ASWebAuthenticationSession(url: url, callbackURLScheme: callbackScheme) {
        [weak self] callbackURL, error in
        if let callbackURL {
          continuation.resume(returning: callbackURL)
        } else {
          continuation.resume(throwing: error ?? ASWebAuthenticationSessionError(.canceledLogin))
        }
        Task { @MainActor in self?.session = nil }
      }
      session.prefersEphemeralWebBrowserSession = true
      session.presentationContextProvider = self
      self.session = session
      if !session.start() {
        self.session = nil
        continuation.resume(throwing: ASWebAuthenticationSessionError(.presentationContextInvalid))
      }
    }
  }
}

extension SystemBrowserSession: ASWebAuthenticationPresentationContextProviding {
  func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
    let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
    return scenes.compactMap(\.keyWindow).first ?? scenes.flatMap(\.windows).first
      ?? ASPresentationAnchor()
  }
}

/// Push needs an APNs key, session-bound device registration and generic
/// attention payloads (PR 1, slice 5). Until then the app only reports the
/// system permission; it neither asks for it nor registers for remote
/// notifications.
enum PushNotifications {
  static let readiness = ProviderReadiness.unavailable(
    "Notifications aren't set up for this experiment yet.")

  nonisolated static func systemPermission() async -> String {
    let status = await UNUserNotificationCenter.current().notificationSettings().authorizationStatus
    switch status {
    case .authorized, .provisional, .ephemeral: return "Allowed in Settings"
    case .denied: return "Off in Settings"
    case .notDetermined: return "Not requested"
    @unknown default: return "Unknown"
    }
  }
}

/// App Store subscriptions need configured products, signed transaction
/// verification and server-side entitlement reconciliation (PR 1, slice 5).
/// The app offers no purchase until a server advertises it.
enum Purchases {
  static let readiness = ProviderReadiness.unavailable(
    "Subscribing in the app isn't available yet. Manage your plan on the web.")

  static var deviceCanPay: Bool { AppStore.canMakePayments }
}
