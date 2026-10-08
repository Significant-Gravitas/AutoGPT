import UIKit

@MainActor
final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene else { return }
    let window = UIWindow(windowScene: windowScene)
    let chat = ChatViewController()
    let navigation = UINavigationController(rootViewController: chat)
    window.rootViewController = navigation
    #if DEBUG
      if ProcessInfo.processInfo.environment["AUTOGPT_UI_TEST_SCREEN"] == "large-status" {
        let category = UIContentSizeCategory.accessibilityExtraExtraExtraLarge
        if #available(iOS 17, *) {
          window.traitOverrides.preferredContentSizeCategory = category
        } else {
          navigation.setOverrideTraitCollection(
            UITraitCollection(preferredContentSizeCategory: category), forChild: chat)
        }
      }
    #endif
    self.window = window
    window.makeKeyAndVisible()
    if let response = connectionOptions.notificationResponse {
      let info = response.notification.request.content.userInfo
      let data: [String: String] = Dictionary(
        uniqueKeysWithValues: ["path", "origin", "binding_id"].compactMap {
          (key: String) -> (String, String)? in
          (info[key] as? String).map { (key, $0) }
        })
      NativePush.shared.receive(data)
    }
  }
}
