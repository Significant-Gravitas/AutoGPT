import Foundation

public struct NavigationPolicy {
  public let origin: URL

  public init(origin: URL) {
    precondition(
      origin.scheme == "https" && origin.host != nil && origin.user == nil && origin.password == nil
    )
    self.origin = origin
  }

  public func isTrusted(_ url: URL) -> Bool {
    url.scheme == "https" && url.host == origin.host
      && (url.port ?? 443) == (origin.port ?? 443)
      && url.user == nil && url.password == nil
  }

  public func isLogin(_ url: URL) -> Bool {
    isTrusted(url) && ["/login", "/signup"].contains(url.path)
  }

  public func isTrustedBlob(_ url: URL) -> Bool {
    guard url.scheme == "blob", let source = URL(string: String(url.absoluteString.dropFirst(5)))
    else { return false }
    return isTrusted(source)
  }
}
