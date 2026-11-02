import Foundation

/// The boundary every network call goes through. `authURL` is the origin of
/// the Better Auth service (the web app's origin on a standard deployment),
/// `apiURL` the origin of the REST backend. The session credential only ever
/// goes to `authURL`; short-lived access tokens only to `apiURL`.
///
/// Discovery from a single entered address is planned server work; until it
/// exists an own server is described by both origins explicitly.
struct ServerConfig: Codable, Hashable, Sendable, Identifiable {
  enum Kind: String, Codable, Sendable { case cloud, custom }

  let id: String
  let name: String
  let authURL: String
  let apiURL: String
  let kind: Kind

  /// Hosted AutoGPT: the web app and its Better Auth routes on
  /// platform.agpt.co, the API on backend.agpt.co. Sign-in only succeeds
  /// once that deployment issues signed native sessions; until then the
  /// sign-in screen reports the server as not yet supporting the app.
  static let cloud = ServerConfig(
    id: "cloud", name: "AutoGPT Cloud", authURL: "https://platform.agpt.co",
    apiURL: "https://backend.agpt.co", kind: .cloud)

  static func custom(name: String, authURL: String, apiURL: String) -> ServerConfig {
    ServerConfig(
      id: "custom-\(Digest.stable("\(authURL)\n\(apiURL)"))",
      name: name.trimmed.isEmpty ? hostLabel(authURL) : name.trimmed,
      authURL: authURL, apiURL: apiURL, kind: .custom)
  }

  /// Partitions credentials and caches: the same origins always share one key.
  var partitionKey: String { Digest.stable("\(authURL)\n\(apiURL)") }

  var authHost: String { Self.hostLabel(authURL) }
  var apiHost: String { Self.hostLabel(apiURL) }

  var isValid: Bool {
    Origin.normalize(authURL) == .valid(authURL) && Origin.normalize(apiURL) == .valid(apiURL)
  }

  static func hostLabel(_ origin: String) -> String {
    guard let components = URLComponents(string: origin), let host = components.host else {
      return origin
    }
    let bare = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
    return components.port.map { "\(bare):\($0)" } ?? bare
  }
}

enum Origin: Equatable {
  case valid(String)
  case invalid(String)

  var failure: String? {
    if case .invalid(let reason) = self { return reason }
    return nil
  }

  /// Accepts `host`, `https://host[:port]` or a trailing slash, and returns a
  /// bare origin. Paths, queries, fragments and embedded credentials are
  /// refused. Plain HTTP is only allowed for this device's loopback
  /// addresses, which is what a simulator reaching a local server uses.
  static func normalize(_ input: String) -> Origin {
    let trimmed = input.trimmed
    guard !trimmed.isEmpty else { return .invalid("Enter an address.") }
    let hasScheme =
      trimmed.range(of: #"^[a-zA-Z][a-zA-Z0-9+.-]*://"#, options: .regularExpression) != nil
    let candidate = hasScheme ? trimmed : "https://\(trimmed)"
    guard let components = URLComponents(string: candidate),
      let scheme = components.scheme?.lowercased()
    else { return .invalid("That isn't a valid web address.") }
    if components.user != nil || components.password != nil {
      return .invalid("Remove the username or password from the address.")
    }
    if !(components.path.isEmpty || components.path == "/") || components.query != nil
      || components.fragment != nil
    {
      return .invalid("Enter only the server address, without a path.")
    }
    let rawHost = components.host ?? ""
    let host = rawHost.trimmingCharacters(in: CharacterSet(charactersIn: "[]")).lowercased()
    guard !host.isEmpty else { return .invalid("That address has no host name.") }
    switch scheme {
    case "http":
      guard isLoopback(host) else {
        return .invalid("Use an https:// address. Plain http only works for localhost.")
      }
    case "https": break
    default: return .invalid("Use an https:// address.")
    }
    let hostPart = host.contains(":") ? "[\(host)]" : host
    let defaultPort = scheme == "https" ? 443 : 80
    let port = components.port.flatMap { $0 == defaultPort ? nil : $0 }
    return .valid("\(scheme)://\(hostPart)\(port.map { ":\($0)" } ?? "")")
  }

  private static func isLoopback(_ host: String) -> Bool {
    host == "localhost" || host == "::1"
      || host.range(of: #"^127(\.\d{1,3}){3}$"#, options: .regularExpression) != nil
  }
}
