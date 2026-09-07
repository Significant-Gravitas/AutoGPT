import AuthenticationServices
import AutoGPTMobileCore
import Foundation
import WebKit

@MainActor
final class NativeAuthentication: NSObject, ASWebAuthenticationPresentationContextProviding {
  private var session: ASWebAuthenticationSession?
  private var exchangeTask: Task<Void, Never>?
  private weak var anchor: UIWindow?
  private var attemptID: UUID?

  func signIn(
    origin: AppOrigin, window: UIWindow, store: WKWebsiteDataStore,
    completion: @escaping @MainActor (Result<Void, Error>) -> Void
  ) {
    cancel()
    anchor = window
    do {
      let pending = try PendingAuthentication()
      let id = UUID()
      attemptID = id
      let authSession = ASWebAuthenticationSession(
        url: pending.startURL(origin: origin), callbackURLScheme: "autogpt"
      ) { [weak self] callback, error in
        Task { @MainActor in
          guard let self, self.attemptID == id else { return }
          self.session = nil
          if let error {
            self.attemptID = nil
            completion(.failure(error))
            return
          }
          guard let callback else {
            self.attemptID = nil
            completion(.failure(MobileError.invalidCallback))
            return
          }
          self.exchangeTask = Task { @MainActor in
            do {
              let code = try pending.validateCallback(callback)
              let cookies = try await self.exchange(code: code, pending: pending, origin: origin)
              guard !Task.isCancelled, self.attemptID == id else { return }
              await store.removeData(
                ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(), modifiedSince: .distantPast)
              guard !Task.isCancelled, self.attemptID == id else { return }
              for cookie in cookies {
                guard !Task.isCancelled, self.attemptID == id else { return }
                await store.httpCookieStore.setCookie(cookie)
              }
              guard !Task.isCancelled, self.attemptID == id else { return }
              self.attemptID = nil
              completion(.success(()))
            } catch {
              guard !Task.isCancelled, self.attemptID == id else { return }
              self.attemptID = nil
              completion(.failure(error))
            }
          }
        }
      }
      authSession.presentationContextProvider = self
      session = authSession
      guard authSession.start() else {
        session = nil
        attemptID = nil
        completion(.failure(MobileError.authenticationFailed))
        return
      }
    } catch { completion(.failure(error)) }
  }

  func cancel() {
    attemptID = nil
    session?.cancel()
    session = nil
    exchangeTask?.cancel()
    exchangeTask = nil
  }

  func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
    anchor ?? ASPresentationAnchor()
  }

  private func exchange(code: String, pending: PendingAuthentication, origin: AppOrigin)
    async throws
    -> [HTTPCookie]
  {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.httpCookieStorage = nil
    configuration.httpShouldSetCookies = false
    configuration.timeoutIntervalForRequest = 30
    configuration.timeoutIntervalForResource = 30
    configuration.urlCache = nil
    let transport = URLSession(
      configuration: configuration, delegate: RejectRedirects(), delegateQueue: nil)
    defer { transport.invalidateAndCancel() }
    var request = URLRequest(url: origin.url.appendingPathComponent("api/auth/mobile/exchange"))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue(origin.url.absoluteString, forHTTPHeaderField: "Origin")
    request.httpBody = try JSONSerialization.data(withJSONObject: [
      "code": code, "code_verifier": pending.verifier,
    ])
    let (_, response) = try await transport.bytes(for: request)
    guard let response = response as? HTTPURLResponse,
      response.statusCode == 200, let responseURL = response.url, origin.contains(responseURL)
    else { throw MobileError.authenticationFailed }
    let fields = response.allHeaderFields.reduce(into: [String: String]()) { output, item in
      guard let name = item.key as? String, let value = item.value as? String else { return }
      output[name] = value
    }
    let cookies = HTTPCookie.cookies(withResponseHeaderFields: fields, for: origin.url)
      .filter { origin.acceptsSessionCookie($0) }
    guard
      cookies.contains(where: {
        $0.name == "better-auth.session_token" || $0.name == "__Secure-better-auth.session_token"
      })
    else { throw MobileError.authenticationFailed }
    return cookies
  }
}

private final class RejectRedirects: NSObject, URLSessionTaskDelegate, Sendable {
  func urlSession(
    _ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
    completionHandler: @escaping @Sendable (URLRequest?) -> Void
  ) {
    completionHandler(nil)
  }
}
