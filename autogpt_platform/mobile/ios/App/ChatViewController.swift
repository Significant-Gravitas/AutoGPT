import AuthenticationServices
import AutoGPTMobileCore
import UIKit
import WebKit

@MainActor
final class ChatViewController: UIViewController {
  private var origin: AppOrigin
  private var webView: WKWebView!
  private let progress = UIProgressView(progressViewStyle: .bar)
  private let statusScroll = UIScrollView()
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
  private var isChangingServer = false
  private var lastCommittedURL: URL?
  private var downloads: [ObjectIdentifier: DownloadExport] = [:]
  private var pushBridge: NativePushBridge?

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
    view.backgroundColor = NativeTheme.background
    view.tintColor = NativeTheme.text
    overrideUserInterfaceStyle = .light
    navigationController?.overrideUserInterfaceStyle = .light
    let appearance = UINavigationBarAppearance()
    appearance.configureWithOpaqueBackground()
    appearance.backgroundColor = NativeTheme.background
    appearance.shadowColor = .clear
    appearance.titleTextAttributes = [
      .font: NativeTheme.font("Geist-Medium", size: 16, style: .headline),
      .foregroundColor: NativeTheme.text,
    ]
    navigationController?.navigationBar.standardAppearance = appearance
    navigationController?.navigationBar.scrollEdgeAppearance = appearance
    navigationController?.navigationBar.compactAppearance = appearance
    navigationController?.navigationBar.tintColor = NativeTheme.text
    progress.progressTintColor = NativeTheme.primary
    navigationController?.navigationBar.prefersLargeTitles = false
    configureStatus()
    installWebView()
    NotificationCenter.default.addObserver(
      self, selector: #selector(openNotification), name: NativePush.opened, object: nil)
    #if DEBUG
      if let screen = ProcessInfo.processInfo.environment["AUTOGPT_UI_TEST_SCREEN"] {
        showSignIn()
        if screen == "error" { showError("Check your connection and try again.") }
        return
      }
    #endif
    loadChat()
    openNotification()
  }

  override func viewWillTransition(
    to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator
  ) {
    view.endEditing(true)
    super.viewWillTransition(to: size, with: coordinator)
  }

  override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
    super.traitCollectionDidChange(previousTraitCollection)
    guard isViewLoaded,
      previousTraitCollection?.preferredContentSizeCategory
        != traitCollection.preferredContentSizeCategory
    else { return }
    updateStatusTypography()
    if let title = statusTitle.text, let message = statusMessage.text {
      applyStatusText(title: title, message: message)
    }
  }

  private func updateStatusTypography() {
    statusTitle.font = NativeTheme.font(
      "Poppins-Medium", size: 28, style: .title1, compatibleWith: traitCollection)
    statusMessage.font = NativeTheme.font(
      "Geist-Regular", size: 14, style: .body, compatibleWith: traitCollection)
    primaryButton.setNeedsUpdateConfiguration()
    secondaryButton.setNeedsUpdateConfiguration()
  }

  private func configureStatus() {
    let mark = UIImageView(image: UIImage(named: "AutoGPTLogo"))
    mark.translatesAutoresizingMaskIntoConstraints = false
    mark.contentMode = .scaleAspectFit
    mark.isAccessibilityElement = false
    let brand = UIView()
    brand.addSubview(mark)
    NSLayoutConstraint.activate([
      brand.heightAnchor.constraint(equalToConstant: 58),
      mark.widthAnchor.constraint(equalToConstant: 128),
      mark.heightAnchor.constraint(equalToConstant: 58),
      mark.topAnchor.constraint(equalTo: brand.topAnchor),
      mark.centerXAnchor.constraint(equalTo: brand.centerXAnchor),
    ])
    status.addArrangedSubview(brand)
    status.axis = .vertical
    status.alignment = .fill
    status.spacing = 16
    status.translatesAutoresizingMaskIntoConstraints = false
    statusTitle.font = NativeTheme.font("Poppins-Medium", size: 28, style: .title1)
    statusTitle.textColor = NativeTheme.text
    statusTitle.adjustsFontForContentSizeCategory = true
    statusTitle.textAlignment = .left
    statusTitle.numberOfLines = 0
    statusTitle.accessibilityTraits.insert(.header)
    statusTitle.accessibilityIdentifier = "Native status title"
    statusMessage.font = NativeTheme.font("Geist-Regular", size: 14, style: .body)
    statusMessage.adjustsFontForContentSizeCategory = true
    statusMessage.textAlignment = .left
    statusMessage.textColor = NativeTheme.secondaryText
    statusMessage.numberOfLines = 0
    statusMessage.accessibilityIdentifier = "Native status message"
    NativeTheme.styleButton(primaryButton, primary: true)
    primaryButton.titleLabel?.adjustsFontForContentSizeCategory = true
    primaryButton.addAction(
      UIAction { [weak self] _ in self?.primaryAction?() }, for: .touchUpInside)
    NativeTheme.styleButton(secondaryButton, primary: false)
    secondaryButton.setTitle("Open in browser", for: .normal)
    secondaryButton.titleLabel?.adjustsFontForContentSizeCategory = true
    secondaryButton.addAction(
      UIAction { [weak self] _ in self?.openInBrowser() }, for: .touchUpInside)
    [statusTitle, statusMessage, primaryButton, secondaryButton].forEach(status.addArrangedSubview)
    status.setCustomSpacing(40, after: brand)
    status.setCustomSpacing(8, after: statusTitle)
    status.setCustomSpacing(32, after: statusMessage)
    status.setCustomSpacing(12, after: primaryButton)
    statusScroll.translatesAutoresizingMaskIntoConstraints = false
    statusScroll.contentInsetAdjustmentBehavior = .never
    statusScroll.alwaysBounceVertical = false
    statusScroll.accessibilityIdentifier = "Native status"
    let content = UIView()
    content.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(statusScroll)
    statusScroll.addSubview(content)
    content.addSubview(status)
    let preferredHeight = content.heightAnchor.constraint(
      equalTo: statusScroll.frameLayoutGuide.heightAnchor)
    preferredHeight.priority = .defaultLow
    let preferredWidth = status.widthAnchor.constraint(
      equalTo: statusScroll.frameLayoutGuide.widthAnchor, constant: -48)
    preferredWidth.priority = .defaultHigh
    NSLayoutConstraint.activate([
      statusScroll.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      statusScroll.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
      statusScroll.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
      statusScroll.bottomAnchor.constraint(equalTo: view.keyboardLayoutGuide.topAnchor),
      content.topAnchor.constraint(equalTo: statusScroll.contentLayoutGuide.topAnchor),
      content.bottomAnchor.constraint(equalTo: statusScroll.contentLayoutGuide.bottomAnchor),
      content.leadingAnchor.constraint(equalTo: statusScroll.contentLayoutGuide.leadingAnchor),
      content.trailingAnchor.constraint(equalTo: statusScroll.contentLayoutGuide.trailingAnchor),
      content.widthAnchor.constraint(equalTo: statusScroll.frameLayoutGuide.widthAnchor),
      content.heightAnchor.constraint(
        greaterThanOrEqualTo: statusScroll.frameLayoutGuide.heightAnchor),
      preferredHeight,
      preferredWidth,
      status.centerYAnchor.constraint(equalTo: content.centerYAnchor),
      status.topAnchor.constraint(greaterThanOrEqualTo: content.topAnchor, constant: 48),
      status.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -48),
      status.leadingAnchor.constraint(greaterThanOrEqualTo: content.leadingAnchor, constant: 24),
      status.trailingAnchor.constraint(lessThanOrEqualTo: content.trailingAnchor, constant: -24),
      status.centerXAnchor.constraint(equalTo: content.centerXAnchor),
      status.widthAnchor.constraint(lessThanOrEqualToConstant: 416),
      primaryButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 46),
      secondaryButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 46),
    ])
  }

  private func installWebView() {
    pushBridge?.invalidate()
    webView?.configuration.userContentController.removeScriptMessageHandler(forName: "AutoGPTPush")
    cancelExports()
    webView?.stopLoading()
    webView?.navigationDelegate = nil
    webView?.uiDelegate = nil
    webView?.removeFromSuperview()
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .default()
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.allowsInlineMediaPlayback = true
    configuration.applicationNameForUserAgent = "AutoGPTMobile/iOS"
    let pushBridge = NativePushBridge(origin: origin)
    configuration.userContentController.add(pushBridge, name: "AutoGPTPush")
    let browser = WKWebView(frame: .zero, configuration: configuration)
    browser.translatesAutoresizingMaskIntoConstraints = false
    browser.navigationDelegate = self
    browser.uiDelegate = self
    browser.allowsBackForwardNavigationGestures = true
    browser.scrollView.contentInsetAdjustmentBehavior = .automatic
    browser.isOpaque = false
    browser.backgroundColor = .systemBackground
    #if DEBUG
      if #available(iOS 16.4, *) { browser.isInspectable = true }
    #endif
    webView = browser
    pushBridge.webView = browser
    self.pushBridge = pushBridge
    view.insertSubview(browser, at: 0)
    NSLayoutConstraint.activate([
      browser.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      browser.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
      browser.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
      browser.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
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
      UIAction(
        title: "Chats, experts & requests",
        image: UIImage(systemName: "bubble.left.and.bubble.right")
      ) {
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
    navigationItem.rightBarButtonItem?.isEnabled = !isSigningIn && !isChangingServer
    navigationItem.leftBarButtonItem?.isEnabled =
      !isSigningIn && !isChangingServer && downloads.isEmpty
  }

  private func loadChat() {
    guard !isSigningIn, !isChangingServer else { return }
    cancelExports()
    statusScroll.isHidden = true
    webView.isHidden = false
    webView.load(URLRequest(url: origin.url.appendingPathComponent("mobile")))
  }

  @objc private func openNotification() {
    guard !isSigningIn, !isChangingServer, let target = NativePush.shared.takeTarget(origin: origin)
    else { return }
    statusScroll.isHidden = true
    webView.isHidden = false
    webView.load(URLRequest(url: target))
  }

  private func clearPush() {
    let binding = NativePush.shared.binding
    if UUID(uuidString: binding) != nil {
      webView?.evaluateJavaScript(
        "fetch('/api/auth/mobile/push/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({binding_id:'\(binding)'})}).catch(()=>{})",
        completionHandler: nil)
    }
    pushBridge?.invalidate()
    NativePush.shared.clear()
  }

  private func retry() {
    guard !isSigningIn, !isChangingServer else { return }
    cancelExports()
    statusScroll.isHidden = true
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
    applyStatusText(title: title, message: message)
    primaryButton.setTitle(button, for: .normal)
    primaryAction = action
    statusScroll.isHidden = false
    statusScroll.setContentOffset(.zero, animated: false)
    UIAccessibility.post(notification: .screenChanged, argument: statusTitle)
  }

  private func applyStatusText(title: String, message: String) {
    updateStatusTypography()
    let headingParagraph = NSMutableParagraphStyle()
    headingParagraph.minimumLineHeight = UIFontMetrics(forTextStyle: .title1).scaledValue(
      for: 40, compatibleWith: traitCollection)
    statusTitle.attributedText = NSAttributedString(
      string: title, attributes: [.paragraphStyle: headingParagraph, .kern: -0.21])
    let bodyParagraph = NSMutableParagraphStyle()
    bodyParagraph.minimumLineHeight = UIFontMetrics(forTextStyle: .body).scaledValue(
      for: 22, compatibleWith: traitCollection)
    statusMessage.attributedText = NSAttributedString(
      string: message, attributes: [.paragraphStyle: bodyParagraph])
  }

  private func showSignIn() {
    clearPush()
    showStatus(
      title: "Welcome back",
      message:
        "Sign in to continue your conversations and run your agents.",
      button: "Sign in to AutoGPT", action: { [weak self] in self?.signIn() })
  }

  private func signIn() {
    guard !isSigningIn, !isChangingServer, let window = view.window else { return }
    clearPush()
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
      case .success:
        self?.lastCommittedURL = nil
        self?.loadChat()
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
    guard !isSigningIn, !isChangingServer, presentedViewController == nil else { return }
    let sheet = ServerSettingsViewController(address: origin.url.absoluteString) {
      [weak self] address in
      guard let self, !self.isSigningIn, !self.isChangingServer else { return }
      do {
        var allowLocalHTTP = false
        #if DEBUG
          allowLocalHTTP = true
        #endif
        let next = try AppOrigin(address, allowLocalHTTP: allowLocalHTTP)
        if next == self.origin { return }
        self.clearPush()
        self.authentication.cancel()
        self.isChangingServer = true
        self.installWebView()
        self.showStatus(
          title: "Changing servers", message: "Clearing the previous website session.",
          button: "Connecting…", action: {})
        self.primaryButton.isEnabled = false
        self.secondaryButton.isEnabled = false
        Task { @MainActor in
          await self.webView.configuration.websiteDataStore.removeData(
            ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(), modifiedSince: .distantPast)
          self.isChangingServer = false
          self.primaryButton.isEnabled = true
          self.secondaryButton.isEnabled = true
          self.origin = next
          self.lastCommittedURL = nil
          UserDefaults.standard.set(next.url.absoluteString, forKey: "serverOrigin")
          self.installWebView()
          self.loadChat()
        }
      } catch { self.showError(error.localizedDescription) }
    }
    present(sheet, animated: true)
  }

}

extension ChatViewController: WKNavigationDelegate {
  func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
    pushBridge?.invalidate()
  }

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
    statusScroll.isHidden = true
    webView.isHidden = false
    updateMenu()
  }

  func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: Error
  ) {
    guard webView === self.webView else { return }
    if !statusScroll.isHidden && webView.isHidden { return }
    let nativeError = error as NSError
    if nativeError.domain == "WebKitErrorDomain" && nativeError.code == 102 { return }
    if nativeError.code != NSURLErrorCancelled {
      showError("Check your connection and try again.")
    }
  }

  func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
    guard webView === self.webView else { return }
    if !statusScroll.isHidden && webView.isHidden { return }
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
    guard webView === self.webView, !isSigningIn, !isChangingServer,
      viewIfLoaded?.window != nil, presentedViewController == nil,
      navigationController?.presentedViewController == nil
    else {
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
    guard webView === self.webView, !isSigningIn, !isChangingServer,
      viewIfLoaded?.window != nil, presentedViewController == nil,
      navigationController?.presentedViewController == nil
    else {
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
