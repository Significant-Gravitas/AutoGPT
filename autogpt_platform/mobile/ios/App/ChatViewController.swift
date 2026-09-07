import AuthenticationServices
import AutoGPTMobileCore
import UIKit
import WebKit

@MainActor
final class ChatViewController: UIViewController {
  private var origin: AppOrigin
  private var webView: WKWebView!
  private let progress = UIProgressView(progressViewStyle: .bar)
  private let status = UIStackView()
  private let statusTitle = UILabel()
  private let statusMessage = UILabel()
  private let primaryButton = UIButton(type: .system)
  private let secondaryButton = UIButton(type: .system)
  private let authentication = NativeAuthentication()
  private var progressObservation: NSKeyValueObservation?
  private var historyObservation: NSKeyValueObservation?
  private var primaryAction: (() -> Void)?
  private var isSigningIn = false
  private var lastCommittedURL: URL?
  private var downloads: [ObjectIdentifier: DownloadExport] = [:]

  init() {
    var address = UserDefaults.standard.string(forKey: "serverOrigin") ?? "https://platform.agpt.co"
    var allowLocalHTTP = false
    #if DEBUG
      allowLocalHTTP = true
      if let override = ProcessInfo.processInfo.environment["AUTOGPT_ORIGIN"] { address = override }
    #endif
    origin =
      (try? AppOrigin(address, allowLocalHTTP: allowLocalHTTP))
      ?? (try! AppOrigin("https://platform.agpt.co"))
    super.init(nibName: nil, bundle: nil)
  }

  required init?(coder: NSCoder) { nil }

  override func viewDidLoad() {
    super.viewDidLoad()
    title = "AutoGPT"
    view.backgroundColor = .systemBackground
    navigationController?.navigationBar.prefersLargeTitles = false
    configureStatus()
    installWebView()
    loadChat()
  }

  private func configureStatus() {
    let mark = UIImageView(image: UIImage(named: "AutoGPTMark"))
    mark.translatesAutoresizingMaskIntoConstraints = false
    mark.contentMode = .scaleAspectFit
    mark.layer.cornerRadius = 16
    mark.clipsToBounds = true
    mark.isAccessibilityElement = false
    let brand = UIView()
    brand.addSubview(mark)
    NSLayoutConstraint.activate([
      brand.heightAnchor.constraint(equalToConstant: 72),
      mark.widthAnchor.constraint(equalToConstant: 64),
      mark.heightAnchor.constraint(equalToConstant: 64),
      mark.topAnchor.constraint(equalTo: brand.topAnchor),
      mark.centerXAnchor.constraint(equalTo: brand.centerXAnchor),
    ])
    status.addArrangedSubview(brand)
    status.axis = .vertical
    status.alignment = .fill
    status.spacing = 16
    status.translatesAutoresizingMaskIntoConstraints = false
    statusTitle.font = .preferredFont(forTextStyle: .title1)
    statusTitle.adjustsFontForContentSizeCategory = true
    statusTitle.textAlignment = .center
    statusTitle.numberOfLines = 0
    statusMessage.font = .preferredFont(forTextStyle: .body)
    statusMessage.adjustsFontForContentSizeCategory = true
    statusMessage.textAlignment = .center
    statusMessage.textColor = .secondaryLabel
    statusMessage.numberOfLines = 0
    primaryButton.configuration = .filled()
    primaryButton.configuration?.cornerStyle = .large
    primaryButton.configuration?.contentInsets = NSDirectionalEdgeInsets(
      top: 16, leading: 20, bottom: 16, trailing: 20)
    primaryButton.addAction(
      UIAction { [weak self] _ in self?.primaryAction?() }, for: .touchUpInside)
    secondaryButton.setTitle("Open in browser", for: .normal)
    secondaryButton.addAction(
      UIAction { [weak self] _ in self?.openInBrowser() }, for: .touchUpInside)
    [statusTitle, statusMessage, primaryButton, secondaryButton].forEach(status.addArrangedSubview)
    view.addSubview(status)
    NSLayoutConstraint.activate([
      status.centerYAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerYAnchor),
      status.leadingAnchor.constraint(
        greaterThanOrEqualTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 28),
      status.trailingAnchor.constraint(
        lessThanOrEqualTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -28),
      status.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      status.widthAnchor.constraint(lessThanOrEqualToConstant: 420),
    ])
  }

  private func installWebView() {
    cancelExports()
    webView?.stopLoading()
    webView?.removeFromSuperview()
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .default()
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.allowsInlineMediaPlayback = true
    configuration.applicationNameForUserAgent = "AutoGPTMobile/iOS"
    let browser = WKWebView(frame: .zero, configuration: configuration)
    browser.translatesAutoresizingMaskIntoConstraints = false
    browser.navigationDelegate = self
    browser.uiDelegate = self
    browser.allowsBackForwardNavigationGestures = true
    browser.scrollView.contentInsetAdjustmentBehavior = .never
    browser.isOpaque = false
    browser.backgroundColor = .systemBackground
    #if DEBUG
      if #available(iOS 16.4, *) { browser.isInspectable = true }
    #endif
    webView = browser
    view.insertSubview(browser, at: 0)
    NSLayoutConstraint.activate([
      browser.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      browser.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
      browser.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
      browser.bottomAnchor.constraint(equalTo: view.keyboardLayoutGuide.topAnchor),
    ])
    if progress.superview == nil {
      progress.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview(progress)
      NSLayoutConstraint.activate([
        progress.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
        progress.leadingAnchor.constraint(equalTo: view.leadingAnchor),
        progress.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      ])
    }
    progressObservation = browser.observe(\.estimatedProgress, options: [.new]) {
      [weak self] browser, _ in
      MainActor.assumeIsolated {
        self?.progress.progress = Float(browser.estimatedProgress)
        self?.progress.isHidden = browser.estimatedProgress >= 1
      }
    }
    historyObservation = browser.observe(\.canGoBack, options: [.new]) { [weak self] _, _ in
      MainActor.assumeIsolated { self?.updateMenu() }
    }
    updateMenu()
  }

  private func updateMenu() {
    navigationItem.leftBarButtonItem =
      webView.canGoBack
      ? UIBarButtonItem(
        image: UIImage(systemName: "chevron.backward"),
        primaryAction: UIAction { [weak self] _ in
          self?.webView.goBack()
        }) : nil
    navigationItem.leftBarButtonItem?.accessibilityLabel = "Back"
    var actions = [
      UIAction(title: "Chat home", image: UIImage(systemName: "bubble.left.and.bubble.right")) {
        [weak self] _ in
        self?.loadChat()
      },
      UIAction(title: "Reload", image: UIImage(systemName: "arrow.clockwise")) { [weak self] _ in
        self?.retry()
      },
      UIAction(title: "Sign in", image: UIImage(systemName: "person.crop.circle")) {
        [weak self] _ in self?.signIn()
      },
      UIAction(title: "Open in browser", image: UIImage(systemName: "safari")) { [weak self] _ in
        self?.openInBrowser()
      },
      UIAction(title: "Server settings", image: UIImage(systemName: "gearshape")) { [weak self] _ in
        self?.settings()
      },
    ]
    if !downloads.isEmpty {
      actions = [
        UIAction(title: "Cancel download", image: UIImage(systemName: "xmark.circle")) {
          [weak self] _ in
          self?.cancelExports()
        }
      ]
      let spinner = UIActivityIndicatorView(style: .medium)
      spinner.startAnimating()
      let label = UILabel()
      label.text = "Preparing file…"
      label.font = .preferredFont(forTextStyle: .subheadline)
      let heading = UIStackView(arrangedSubviews: [spinner, label])
      heading.axis = .horizontal
      heading.spacing = 8
      navigationItem.titleView = heading
    } else {
      navigationItem.titleView = nil
    }
    navigationItem.rightBarButtonItem = UIBarButtonItem(
      image: UIImage(systemName: "ellipsis.circle"), menu: UIMenu(children: actions))
    navigationItem.rightBarButtonItem?.accessibilityLabel = "App menu"
    navigationItem.rightBarButtonItem?.isEnabled = !isSigningIn
    navigationItem.leftBarButtonItem?.isEnabled = !isSigningIn && downloads.isEmpty
  }

  private func loadChat() {
    cancelExports()
    status.isHidden = true
    webView.isHidden = false
    webView.load(URLRequest(url: origin.chatURL))
  }

  private func retry() {
    cancelExports()
    status.isHidden = true
    webView.isHidden = false
    let target = lastCommittedURL.flatMap { origin.contains($0) ? $0 : nil } ?? origin.chatURL
    webView.load(URLRequest(url: target))
  }

  private func showStatus(
    title: String, message: String, button: String, action: @escaping () -> Void
  ) {
    view.endEditing(true)
    webView.isHidden = true
    progress.isHidden = true
    statusTitle.text = title
    statusMessage.text = message
    primaryButton.setTitle(button, for: .normal)
    primaryAction = action
    status.isHidden = false
  }

  private func showSignIn() {
    showStatus(
      title: "Your AutoGPT, on the go",
      message:
        "Sign in securely to continue your conversations, run agents, and pick up where you left off.",
      button: "Sign in to AutoGPT", action: { [weak self] in self?.signIn() })
  }

  private func signIn() {
    guard !isSigningIn, let window = view.window else { return }
    isSigningIn = true
    installWebView()
    showStatus(
      title: "Signing in", message: "Finish signing in through your secure browser.",
      button: "Signing in…", action: {})
    primaryButton.isEnabled = false
    secondaryButton.isEnabled = false
    authentication.signIn(
      origin: origin, window: window,
      store: webView.configuration.websiteDataStore
    ) { [weak self] result in
      self?.isSigningIn = false
      self?.primaryButton.isEnabled = true
      self?.secondaryButton.isEnabled = true
      self?.updateMenu()
      switch result {
      case .success: self?.loadChat()
      case .failure(let error):
        if (error as? ASWebAuthenticationSessionError)?.code == .canceledLogin
          || (error as? MobileError) == .authenticationCancelled
        {
          self?.showSignIn()
        } else {
          self?.showError(error.localizedDescription)
        }
      }
    }
  }

  private func showError(_ message: String) {
    showStatus(
      title: "Let's reconnect", message: message, button: "Try again",
      action: { [weak self] in self?.retry() })
  }

  private func openExternal(_ url: URL, directly: Bool) {
    guard URLComponents(url: url, resolvingAgainstBaseURL: false)?.user == nil else { return }
    if directly {
      UIApplication.shared.open(url)
      return
    }
    guard presentedViewController == nil else { return }
    let alert = UIAlertController(
      title: "Open in your browser?",
      message:
        "This page wants to open \(url.host ?? "another app"). Your AutoGPT conversation will stay here.",
      preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
    alert.addAction(
      UIAlertAction(title: "Open", style: .default) { _ in UIApplication.shared.open(url) })
    present(alert, animated: true)
  }

  private func openInBrowser() {
    let target = webView.url.flatMap { origin.contains($0) ? $0 : nil } ?? origin.chatURL
    UIApplication.shared.open(target)
  }

  private func settings() {
    let alert = UIAlertController(
      title: "Server settings",
      message:
        "Connect to AutoGPT or your own HTTPS server. Changing servers clears this app's saved website session.",
      preferredStyle: .alert)
    alert.addTextField { [origin] field in
      field.text = origin.url.absoluteString
      field.placeholder = "https://platform.agpt.co"
      field.accessibilityLabel = "Server address"
      field.keyboardType = .URL
      field.autocapitalizationType = .none
      field.autocorrectionType = .no
    }
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
    alert.addAction(
      UIAlertAction(title: "Connect", style: .default) { [weak self, weak alert] _ in
        guard let self, let address = alert?.textFields?.first?.text else { return }
        do {
          var allowLocalHTTP = false
          #if DEBUG
            allowLocalHTTP = true
          #endif
          let next = try AppOrigin(address, allowLocalHTTP: allowLocalHTTP)
          if next == self.origin { return }
          self.authentication.cancel()
          self.webView.stopLoading()
          Task { @MainActor in
            await self.webView.configuration.websiteDataStore.removeData(
              ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(), modifiedSince: .distantPast)
            self.origin = next
            self.lastCommittedURL = nil
            UserDefaults.standard.set(next.url.absoluteString, forKey: "serverOrigin")
            self.installWebView()
            self.loadChat()
          }
        } catch { self.showError(error.localizedDescription) }
      })
    present(alert, animated: true)
  }
}

extension ChatViewController: WKNavigationDelegate {
  func webView(
    _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
    decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
  ) {
    guard webView === self.webView else {
      decisionHandler(.cancel)
      return
    }
    guard let url = navigationAction.request.url else {
      decisionHandler(.cancel)
      return
    }
    if navigationAction.shouldPerformDownload
      && (origin.contains(url) || origin.allowsBlobDownload(url))
    {
      decisionHandler(.download)
      return
    }
    if navigationAction.targetFrame?.isMainFrame == false {
      decisionHandler(
        ["https", "about"].contains(url.scheme) || origin.contains(url)
          || origin.allowsBlobDownload(url) ? .allow : .cancel)
      return
    }
    if origin.contains(url) {
      if url.path == "/login" || url.path == "/auth/login" {
        decisionHandler(.cancel)
        showSignIn()
      } else if navigationAction.shouldPerformDownload {
        decisionHandler(.download)
      } else {
        decisionHandler(.allow)
      }
    } else if navigationAction.shouldPerformDownload && origin.allowsBlobDownload(url) {
      decisionHandler(.download)
    } else {
      decisionHandler(.cancel)
      if ["https", "http", "mailto", "tel"].contains(url.scheme) {
        openExternal(url, directly: navigationAction.navigationType == .linkActivated)
      }
    }
  }

  func webView(
    _ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
    decisionHandler: @escaping @MainActor @Sendable (WKNavigationResponsePolicy) -> Void
  ) {
    guard webView === self.webView else {
      decisionHandler(.cancel)
      return
    }
    if navigationResponse.isForMainFrame,
      let response = navigationResponse.response as? HTTPURLResponse, response.statusCode >= 400
    {
      decisionHandler(.cancel)
      showError("The server returned an error (\(response.statusCode)). Try again in a moment.")
    } else {
      decisionHandler(navigationResponse.canShowMIMEType ? .allow : .download)
    }
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    guard webView === self.webView else { return }
    if let url = webView.url, origin.contains(url) { lastCommittedURL = url }
    status.isHidden = true
    webView.isHidden = false
    updateMenu()
  }

  func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: Error
  ) {
    guard webView === self.webView else { return }
    if !status.isHidden && webView.isHidden { return }
    let nativeError = error as NSError
    if nativeError.domain == "WebKitErrorDomain" && nativeError.code == 102 { return }
    if nativeError.code != NSURLErrorCancelled {
      showError("Check your connection and try again.")
    }
  }

  func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
    guard webView === self.webView else { return }
    if !status.isHidden && webView.isHidden { return }
    let nativeError = error as NSError
    if nativeError.domain == "WebKitErrorDomain" && nativeError.code == 102 { return }
    if nativeError.code != NSURLErrorCancelled {
      showError("The page stopped loading. Please try again.")
    }
  }

  func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    guard webView === self.webView else { return }
    installWebView()
    showError("The page was paused to free memory. Reconnect to continue your conversation.")
  }

  func webView(
    _ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload
  ) {
    guard webView === self.webView else {
      Task { _ = await download.cancel() }
      return
    }
    export(download)
  }

  func webView(
    _ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload
  ) {
    guard webView === self.webView else {
      Task { _ = await download.cancel() }
      return
    }
    export(download)
  }

  private func export(_ download: WKDownload) {
    guard downloads.isEmpty else {
      Task { _ = await download.cancel() }
      return
    }
    let key = ObjectIdentifier(download)
    let exporter = DownloadExport(
      presenter: self, origin: origin, download: download,
      completion: { [weak self] in
        self?.downloads.removeValue(forKey: key)
        self?.updateMenu()
      },
      onError: { [weak self] message in
        guard let self, self.presentedViewController == nil else { return }
        let alert = UIAlertController(
          title: "Couldn't save file", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        self.present(alert, animated: true)
      })
    downloads[key] = exporter
    download.delegate = exporter
    updateMenu()
  }

  private func cancelExports() {
    let pending = Array(downloads.values)
    downloads.removeAll()
    for export in pending { export.cancel() }
    if webView != nil { updateMenu() }
  }

}

extension ChatViewController: WKUIDelegate {
  func webView(
    _ webView: WKWebView,
    requestMediaCapturePermissionFor securityOrigin: WKSecurityOrigin,
    initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
    decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
  ) {
    guard webView === self.webView else {
      decisionHandler(.deny)
      return
    }
    var source = URLComponents()
    source.scheme = securityOrigin.protocol
    source.host = securityOrigin.host
    source.port = securityOrigin.port == 0 ? nil : securityOrigin.port
    guard frame.isMainFrame, type == .microphone,
      let url = source.url, origin.contains(url)
    else {
      decisionHandler(.deny)
      return
    }
    decisionHandler(.prompt)
  }

  func webView(
    _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
    for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures
  ) -> WKWebView? {
    guard webView === self.webView else { return nil }
    guard navigationAction.targetFrame == nil, let url = navigationAction.request.url else {
      return nil
    }
    if origin.contains(url) {
      webView.load(navigationAction.request)
    } else if ["https", "http"].contains(url.scheme) {
      openExternal(url, directly: false)
    }
    return nil
  }

  func webView(
    _ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable () -> Void
  ) {
    guard webView === self.webView else {
      completionHandler()
      return
    }
    let alert = UIAlertController(title: "AutoGPT", message: message, preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler() })
    present(alert, animated: true)
  }

  func webView(
    _ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
  ) {
    guard webView === self.webView else {
      completionHandler(false)
      return
    }
    let alert = UIAlertController(title: "AutoGPT", message: message, preferredStyle: .alert)
    alert.addAction(
      UIAlertAction(title: "Cancel", style: .cancel) { _ in completionHandler(false) })
    alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler(true) })
    present(alert, animated: true)
  }
}
