import UIKit
import WebKit

extension ShellViewController: WKNavigationDelegate, WKUIDelegate {
  func webView(
    _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    guard let url = action.request.url else {
      decisionHandler(.cancel)
      return
    }
    if action.targetFrame?.isMainFrame == false {
      decisionHandler(.allow)
      return
    }
    if policy.isLogin(url) {
      decisionHandler(.cancel)
      showMessage("Sign in to use AutoGPT.")
      signIn()
    } else if policy.isTrusted(url) {
      decisionHandler(action.shouldPerformDownload ? .download : .allow)
    } else {
      decisionHandler(.cancel)
      if ["https", "http"].contains(url.scheme) { UIApplication.shared.open(url) }
    }
  }

  func webView(
    _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
    for action: WKNavigationAction, windowFeatures: WKWindowFeatures
  ) -> WKWebView? {
    guard let url = action.request.url else { return nil }
    if ["https", "http"].contains(url.scheme) {
      UIApplication.shared.open(url)
    }
    return nil
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { showMessage(nil) }

  func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: Error
  ) {
    if (error as NSError).code != NSURLErrorCancelled {
      showMessage("Could not connect. Check your connection and tap Refresh.")
    }
  }

  func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
    self.webView(webView, didFailProvisionalNavigation: navigation, withError: error)
  }

  func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { goHome() }

  func webView(
    _ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
    decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
  ) {
    if let http = response.response as? HTTPURLResponse, http.statusCode >= 400,
      response.isForMainFrame
    {
      showMessage("AutoGPT could not load this page. Tap Sign in or Refresh to try again.")
      decisionHandler(.cancel)
      return
    }
    decisionHandler(response.canShowMIMEType ? .allow : .download)
  }

  func webView(
    _ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload
  ) { download.delegate = self }
  func webView(
    _ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload
  ) { download.delegate = self }
}

extension ShellViewController: WKDownloadDelegate {
  func download(
    _ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
    completionHandler: @escaping (URL?) -> Void
  ) {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    do {
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      let url = folder.appendingPathComponent(
        URL(fileURLWithPath: suggestedFilename).lastPathComponent)
      downloads[ObjectIdentifier(download)] = url
      completionHandler(url)
    } catch { completionHandler(nil) }
  }

  func downloadDidFinish(_ download: WKDownload) {
    guard let url = downloads.removeValue(forKey: ObjectIdentifier(download)) else { return }
    let share = UIActivityViewController(activityItems: [url], applicationActivities: nil)
    share.popoverPresentationController?.sourceView = view
    share.completionWithItemsHandler = { _, _, _, _ in
      try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }
    present(share, animated: true)
  }

  func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
    if let url = downloads.removeValue(forKey: ObjectIdentifier(download)) {
      try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }
    showMessage("Download failed. Please try again.")
  }
}
