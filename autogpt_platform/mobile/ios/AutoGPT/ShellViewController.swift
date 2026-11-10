import AuthenticationServices
import UIKit
import WebKit

final class ShellViewController: UIViewController, ASWebAuthenticationPresentationContextProviding {
  let policy = NavigationPolicy(
    origin: URL(string: Bundle.main.object(forInfoDictionaryKey: "AutoGPTOrigin") as! String)!)
  let webView: WKWebView = {
    let configuration = WKWebViewConfiguration()
    configuration.applicationNameForUserAgent = "AutoGPTMobile/0.1 iOS"
    return WKWebView(frame: .zero, configuration: configuration)
  }()
  private let message = UILabel()
  private let spinner = UIActivityIndicatorView(style: .medium)
  private var authSession: ASWebAuthenticationSession?
  private var progressObservation: NSKeyValueObservation?
  var downloads: [ObjectIdentifier: URL] = [:]

  override func viewDidLoad() {
    super.viewDidLoad()
    title = "AutoGPT"
    view.backgroundColor = .systemBackground
    webView.navigationDelegate = self
    webView.uiDelegate = self
    webView.allowsBackForwardNavigationGestures = true
    message.numberOfLines = 0
    message.textAlignment = .center
    message.isHidden = true
    let stack = UIStackView(arrangedSubviews: [message, webView])
    stack.axis = .vertical
    stack.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(stack)
    NSLayoutConstraint.activate([
      stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      stack.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
      stack.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      stack.trailingAnchor.constraint(equalTo: view.trailingAnchor),
    ])
    navigationItem.leftBarButtonItems = [
      UIBarButtonItem(
        image: UIImage(systemName: "chevron.left"), style: .plain, target: self,
        action: #selector(goBack)),
      UIBarButtonItem(
        image: UIImage(systemName: "house"), style: .plain, target: self, action: #selector(goHome)),
    ]
    navigationItem.rightBarButtonItems = [
      UIBarButtonItem(title: "Sign in", style: .plain, target: self, action: #selector(signIn)),
      UIBarButtonItem(barButtonSystemItem: .refresh, target: self, action: #selector(reload)),
      UIBarButtonItem(customView: spinner),
    ]
    navigationItem.leftBarButtonItems?[0].accessibilityLabel = "Back"
    navigationItem.leftBarButtonItems?[1].accessibilityLabel = "Home"
    progressObservation = webView.observe(\.isLoading, options: [.new]) { [weak self] _, change in
      if change.newValue == true {
        self?.spinner.startAnimating()
      } else {
        self?.spinner.stopAnimating()
      }
    }
    goHome()
  }

  @objc private func goBack() { if webView.canGoBack { webView.goBack() } }
  @objc func goHome() {
    webView.load(URLRequest(url: policy.origin.appendingPathComponent("home")))
  }
  @objc private func reload() {
    if let url = webView.url, policy.isTrusted(url), !url.path.hasPrefix("/api/auth/mobile/") {
      webView.reload()
    } else {
      goHome()
    }
  }

  @objc func signIn() {
    guard authSession == nil else { return }
    let attempt = AuthAttempt()
    authSession = ASWebAuthenticationSession(
      url: attempt.startURL(origin: policy.origin), callbackURLScheme: "autogpt"
    ) { [weak self] url, _ in
      DispatchQueue.main.async {
        guard let self else { return }
        self.authSession = nil
        guard let url, let body = attempt.exchangeBody(callback: url) else {
          self.showMessage("Sign-in did not finish. Tap Sign in to try again.")
          return
        }
        var request = URLRequest(
          url: self.policy.origin.appendingPathComponent("api/auth/mobile/exchange"))
        request.httpMethod = "POST"
        request.httpBody = Data(body.utf8)
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue(self.policy.origin.absoluteString, forHTTPHeaderField: "Origin")
        self.webView.load(request)
      }
    }
    authSession?.presentationContextProvider = self
    if authSession?.start() != true {
      authSession = nil
      showMessage("Could not open sign-in. Please try again.")
    }
  }

  func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
    view.window!
  }

  func showMessage(_ text: String?) {
    message.text = text
    message.isHidden = text == nil
  }
}
