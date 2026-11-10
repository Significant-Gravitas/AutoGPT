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
  private let retry = UIButton(configuration: .plain())
  private var authSession: ASWebAuthenticationSession?
  var downloads: [ObjectIdentifier: URL] = [:]

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .systemBackground
    webView.navigationDelegate = self
    webView.uiDelegate = self
    webView.allowsBackForwardNavigationGestures = true
    message.numberOfLines = 0
    message.textAlignment = .center
    message.isHidden = true
    retry.setTitle("Try again", for: .normal)
    retry.isHidden = true
    retry.addAction(
      UIAction { [weak self] _ in
        self?.showMessage(nil)
        self?.goHome()
      }, for: .touchUpInside)
    let notice = UIStackView(arrangedSubviews: [message, retry])
    notice.axis = .vertical
    notice.backgroundColor = .systemBackground
    notice.translatesAutoresizingMaskIntoConstraints = false
    webView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(webView)
    view.addSubview(notice)
    NSLayoutConstraint.activate([
      webView.topAnchor.constraint(equalTo: view.topAnchor),
      webView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
      webView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      webView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      notice.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      notice.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      notice.trailingAnchor.constraint(equalTo: view.trailingAnchor),
    ])
    goHome()
  }

  func goHome() {
    webView.load(URLRequest(url: policy.origin.appendingPathComponent("home")))
  }

  func signIn() {
    guard authSession == nil else { return }
    let attempt = AuthAttempt()
    // Only an app with this host as an associated domain can claim the callback.
    authSession = ASWebAuthenticationSession(
      url: attempt.startURL(origin: policy.origin),
      callback: .https(host: policy.origin.host!, path: AuthAttempt.callbackPath)
    ) { [weak self] url, _ in
      DispatchQueue.main.async {
        guard let self else { return }
        self.authSession = nil
        guard let url, let body = attempt.exchangeBody(callback: url, policy: self.policy) else {
          self.showMessage("Sign-in did not finish.", canRetry: true)
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
      showMessage("Could not open sign-in.", canRetry: true)
    }
  }

  func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
    view.window!
  }

  func showMessage(_ text: String?, canRetry: Bool = false) {
    message.text = text
    message.isHidden = text == nil
    retry.isHidden = !canRetry
  }
}
