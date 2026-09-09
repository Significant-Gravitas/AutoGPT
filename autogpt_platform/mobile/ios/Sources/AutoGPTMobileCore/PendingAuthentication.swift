import CryptoKit
import Foundation
import Security

public struct PendingAuthentication: Sendable {
  public let verifier: String
  public let state: String
  public let createdAt: Date
  public var challenge: String { Self.challenge(for: verifier) }

  public init(now: Date = Date()) throws {
    verifier = try Self.randomToken()
    state = try Self.randomToken()
    createdAt = now
  }

  public func startURL(origin: AppOrigin) -> URL {
    var parts = URLComponents(
      url: origin.url.appendingPathComponent("api/auth/mobile/start"),
      resolvingAgainstBaseURL: false)!
    parts.queryItems = [
      URLQueryItem(name: "code_challenge", value: challenge),
      URLQueryItem(name: "state", value: state),
    ]
    return parts.url!
  }

  public func validateCallback(_ url: URL, now: Date = Date()) throws -> String {
    guard now.timeIntervalSince(createdAt) >= 0, now.timeIntervalSince(createdAt) < 600
    else { throw MobileError.expiredAuthentication }
    guard let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
      parts.scheme == "autogpt", parts.host == "auth", parts.path == "/callback",
      parts.user == nil, parts.password == nil, parts.port == nil, parts.fragment == nil,
      let items = parts.queryItems,
      items.filter({ $0.name == "state" }).count == 1,
      items.first(where: { $0.name == "state" })?.value == state
    else { throw MobileError.invalidCallback }
    if items.filter({ $0.name == "error" }).count == 1,
      items.first(where: { $0.name == "error" })?.value == "access_denied",
      !items.contains(where: { $0.name == "code" })
    {
      throw MobileError.authenticationCancelled
    }
    guard items.filter({ $0.name == "code" }).count == 1,
      !items.contains(where: { $0.name == "error" }),
      let code = items.first(where: { $0.name == "code" })?.value,
      (32...128).contains(code.count),
      code.range(of: "^[A-Za-z0-9_-]{32,128}$", options: .regularExpression) != nil
    else { throw MobileError.invalidCallback }
    return code
  }

  public static func challenge(for verifier: String) -> String {
    base64URL(Data(SHA256.hash(data: Data(verifier.utf8))))
  }

  private static func randomToken() throws -> String {
    var bytes = [UInt8](repeating: 0, count: 32)
    guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess
    else { throw MobileError.randomGeneration }
    return base64URL(Data(bytes))
  }

  private static func base64URL(_ data: Data) -> String {
    data.base64EncodedString().replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
  }
}
