import AutoGPTMobileCore
import WebKit

@MainActor
final class NativePushBridge: NSObject, WKScriptMessageHandler {
  let origin: AppOrigin
  weak var webView: WKWebView?
  private var generation = 0

  init(origin: AppOrigin) { self.origin = origin }

  func invalidate() { generation += 1 }

  func userContentController(
    _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
  ) {
    guard message.frameInfo.isMainFrame, let url = message.frameInfo.request.url,
      origin.contains(url),
      let raw = message.body as? String, raw.utf8.count < 2048, let data = raw.data(using: .utf8),
      let request = try? JSONSerialization.jsonObject(with: data) as? [String: String],
      let id = request["id"], UUID(uuidString: id) != nil,
      let action = request["action"], ["status", "enable", "disable"].contains(action),
      let account = request["account_id"], account.count <= 128
    else { return }
    let current = generation
    Task { @MainActor [weak self] in
      guard let self else { return }
      var result = await NativePush.shared.handle(action: action, account: account, origin: origin)
      guard generation == current, let webView, let currentURL = webView.url,
        origin.contains(currentURL)
      else { return }
      result["id"] = id
      guard let json = try? JSONSerialization.data(withJSONObject: result),
        let text = String(data: json, encoding: .utf8)
      else { return }
      webView.evaluateJavaScript(
        "window.dispatchEvent(new CustomEvent('autogpt-native-push', {detail: \(text)}))",
        completionHandler: nil)
    }
  }
}
