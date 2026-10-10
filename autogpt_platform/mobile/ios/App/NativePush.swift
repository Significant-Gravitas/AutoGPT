import AutoGPTMobileCore
import UIKit
import UserNotifications

@MainActor
final class NativePush: NSObject, UNUserNotificationCenterDelegate {
  static let shared = NativePush()
  static let opened = Notification.Name("AutoGPTPushOpened")
  private let defaults = UserDefaults.standard
  private var token: String?
  private var callbacks: [(String?) -> Void] = []
  private var generation = 0
  private(set) var pendingTarget: [String: String]?

  var binding: String { defaults.string(forKey: "pushBinding") ?? "" }

  func clear() {
    generation += 1
    defaults.removeObject(forKey: "pushBinding")
    defaults.set(false, forKey: "pushEnabled")
    pendingTarget = nil
    UNUserNotificationCenter.current().removeAllDeliveredNotifications()
  }

  func handle(action: String, account: String, origin: AppOrigin) async -> [String: String] {
    if defaults.string(forKey: "pushAccount") != account
      || defaults.string(forKey: "pushOrigin") != origin.url.absoluteString
    {
      clear()
      defaults.set(account, forKey: "pushAccount")
      defaults.set(origin.url.absoluteString, forKey: "pushOrigin")
    }
    if action == "disable" || account.isEmpty {
      clear()
      return ["permission": "disabled"]
    }
    if action == "status" && !defaults.bool(forKey: "pushEnabled") {
      return ["permission": "disabled"]
    }
    let current = generation
    let center = UNUserNotificationCenter.current()
    if action == "enable" {
      guard (try? await center.requestAuthorization(options: [.alert, .badge, .sound])) == true
      else { return ["permission": "denied"] }
    }
    guard await hasPermission(center) else { return ["permission": "denied"] }
    let expectedAccount = account
    let registered = await registrationToken()
    guard current == generation, defaults.string(forKey: "pushAccount") == expectedAccount,
      defaults.string(forKey: "pushOrigin") == origin.url.absoluteString, let registered
    else { return ["permission": "unavailable"] }
    if binding.isEmpty { defaults.set(UUID().uuidString, forKey: "pushBinding") }
    defaults.set(true, forKey: "pushEnabled")
    let environment =
      Bundle.main.object(forInfoDictionaryKey: "AutoGPTPushEnvironment") as? String ?? "sandbox"
    return [
      "permission": "granted", "provider": "apns", "environment": environment, "token": registered,
      "binding_id": binding,
    ]
  }

  private func registrationToken() async -> String? {
    if let token { return token }
    return await withCheckedContinuation { continuation in
      callbacks.append { continuation.resume(returning: $0) }
      UIApplication.shared.registerForRemoteNotifications()
      Task { @MainActor in
        try? await Task.sleep(for: .seconds(15))
        if !callbacks.isEmpty { registered(nil) }
      }
    }
  }

  private func hasPermission(_ center: UNUserNotificationCenter) async -> Bool {
    await withCheckedContinuation { continuation in
      center.getNotificationSettings { settings in
        continuation.resume(
          returning: settings.authorizationStatus == .authorized
            || settings.authorizationStatus == .provisional)
      }
    }
  }

  func registered(_ data: Data?) {
    token = data?.map { String(format: "%02x", $0) }.joined()
    let waiting = callbacks
    callbacks.removeAll()
    for callback in waiting { callback(token) }
  }

  func takeTarget(origin: AppOrigin) -> URL? {
    guard let target = pendingTarget else { return nil }
    pendingTarget = nil
    return NotificationTarget.url(
      path: target["path"] ?? "", notificationOrigin: target["origin"] ?? "",
      binding: target["binding_id"] ?? "", expectedBinding: binding, origin: origin)
  }

  func receive(_ data: [String: String]) {
    guard defaults.bool(forKey: "pushEnabled"), data["binding_id"] == binding else { return }
    pendingTarget = data
    NotificationCenter.default.post(name: Self.opened, object: nil)
  }

  nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
    withCompletionHandler completionHandler: @escaping () -> Void
  ) {
    let data = response.notification.request.content.userInfo.compactMapValues { $0 as? String }
    let values = Dictionary(
      uniqueKeysWithValues: data.compactMap { key, value in (key as? String).map { ($0, value) } })
    Task { @MainActor in self.receive(values) }
    completionHandler()
  }

  nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, willPresent notification: UNNotification,
    withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
  ) {
    completionHandler([.banner, .sound])
  }
}
