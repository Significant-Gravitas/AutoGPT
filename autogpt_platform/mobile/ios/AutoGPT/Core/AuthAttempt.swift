import CryptoKit
import Foundation

public struct AuthAttempt {
  public let verifier: String
  public let state: String
  private let createdAt: Date

  public init(createdAt: Date = Date()) {
    self.createdAt = createdAt
    verifier = Self.randomToken()
    state = Self.randomToken()
  }

  public func startURL(origin: URL) -> URL {
    var url = URLComponents(
      url: origin.appendingPathComponent("api/auth/mobile/start"), resolvingAgainstBaseURL: false)!
    let challenge = Self.base64url(Data(SHA256.hash(data: Data(verifier.utf8))))
    url.queryItems = [
      URLQueryItem(name: "challenge", value: challenge), URLQueryItem(name: "state", value: state),
    ]
    return url.url!
  }

  public func exchangeBody(callback: URL) -> String? {
    guard Date().timeIntervalSince(createdAt) < 600,
      callback.scheme == "autogpt", callback.host == "auth",
      callback.path == "/callback", callback.user == nil,
      callback.password == nil, callback.port == nil, callback.fragment == nil,
      let items = URLComponents(url: callback, resolvingAgainstBaseURL: false)?.queryItems,
      items.count == 2,
      items.filter({ $0.name == "state" }).count == 1,
      items.first(where: { $0.name == "state" })?.value == state,
      let code = items.first(where: { $0.name == "code" })?.value,
      code.range(of: "^[A-Za-z0-9_-]{43}$", options: .regularExpression) != nil
    else { return nil }
    return "code=\(code)&verifier=\(verifier)"
  }

  private static func randomToken() -> String {
    var generator = SystemRandomNumberGenerator()
    return base64url(Data((0..<32).map { _ in UInt8.random(in: .min ... .max, using: &generator) }))
  }

  private static func base64url(_ data: Data) -> String {
    data.base64EncodedString().replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
  }
}
