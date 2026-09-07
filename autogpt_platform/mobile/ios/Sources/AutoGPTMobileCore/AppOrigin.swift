import Foundation

public struct AppOrigin: Equatable, Sendable {
  public let url: URL

  public init(_ value: String, allowLocalHTTP: Bool = false) throws {
    guard var parts = URLComponents(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
      let host = parts.host?.lowercased(), !host.isEmpty,
      parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
      parts.path.isEmpty || parts.path == "/",
      parts.port == nil || (1...65535).contains(parts.port!),
      parts.scheme?.lowercased() == "https"
        || (allowLocalHTTP && parts.scheme?.lowercased() == "http"
          && ["localhost", "127.0.0.1", "[::1]", "::1"].contains(host))
    else { throw MobileError.invalidOrigin }
    parts.scheme = parts.scheme?.lowercased()
    parts.host = host
    parts.path = ""
    if parts.port == (parts.scheme == "https" ? 443 : 80) { parts.port = nil }
    guard let normalized = parts.url else { throw MobileError.invalidOrigin }
    url = normalized
  }

  public var chatURL: URL { url.appendingPathComponent("copilot") }

  public func contains(_ candidate: URL) -> Bool {
    guard let parts = URLComponents(url: candidate, resolvingAgainstBaseURL: false),
      parts.user == nil, parts.password == nil
    else { return false }
    return parts.scheme?.lowercased() == url.scheme
      && parts.host?.lowercased() == url.host?.lowercased()
      && (parts.port ?? (parts.scheme == "https" ? 443 : 80))
        == (url.port ?? (url.scheme == "https" ? 443 : 80))
  }

  public func allowsBlobDownload(_ candidate: URL) -> Bool {
    guard candidate.scheme == "blob",
      let source = URL(string: String(candidate.absoluteString.dropFirst(5)))
    else { return false }
    return contains(source)
  }

  public func acceptsSessionCookie(_ cookie: HTTPCookie, now: Date = Date()) -> Bool {
    cookie.domain.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
      == url.host?.lowercased()
      && (url.scheme != "https" || cookie.isSecure)
      && cookie.isHTTPOnly && cookie.path == "/"
      && !cookie.value.isEmpty && (cookie.expiresDate == nil || cookie.expiresDate! > now)
  }

  public func validatedSessionCookies(_ cookies: [HTTPCookie], now: Date = Date()) -> [HTTPCookie]?
  {
    let tokens = cookies.filter {
      $0.name == "better-auth.session_token" || $0.name == "__Secure-better-auth.session_token"
    }
    guard tokens.count == 1, acceptsSessionCookie(tokens[0], now: now) else { return nil }
    return cookies.filter { acceptsSessionCookie($0, now: now) }
  }
}

public enum MobileError: Error, LocalizedError {
  case invalidOrigin, randomGeneration, invalidCallback, expiredAuthentication,
    authenticationFailed, authenticationCancelled

  public var errorDescription: String? {
    switch self {
    case .invalidOrigin: "Enter an HTTPS address without a path, password, or query."
    case .randomGeneration: "A secure sign-in request could not be created. Please try again."
    case .invalidCallback: "The sign-in response did not match this app. Please start again."
    case .expiredAuthentication: "This sign-in request expired. Please start again."
    case .authenticationCancelled: "Sign-in was canceled."
    case .authenticationFailed:
      "Sign-in could not be completed. Check that this server supports the mobile app."
    }
  }
}
