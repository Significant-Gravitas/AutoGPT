import Foundation

struct SignInError: UserFacingError {
  enum Kind: Sendable {
    case credentials, unverified, blocked, rateLimited, incompatible, rejected, unavailable
  }

  let kind: Kind
  let message: String

  init(_ kind: Kind, _ message: String) {
    self.kind = kind
    self.message = message
  }

  var userMessage: String { message }
}

struct SignedInUser: Sendable {
  let sessionToken: String
  let userID: String
  let email: String
  let name: String?
}

struct AccessGrant: Sendable {
  let token: String
  let expiresAt: Date
  /// A re-issued signed session credential, when the auth server sends one.
  let renewedSessionToken: String?
}

/// The server's own Better Auth routes. The signed session credential only
/// ever arrives in the `set-auth-token` header and only goes to the auth origin.
enum AuthAPI {
  /// Email sign-in through the server's Better Auth policies. The token in
  /// the JSON body is unsigned and is deliberately ignored.
  static func signIn(server: ServerConfig, email: String, password: String) async throws
    -> SignedInUser
  {
    let request = HTTPRequest(
      url: try endpoint(server, "/api/auth/sign-in/email"), method: "POST",
      json: .object([
        "email": .string(email), "password": .string(password), "rememberMe": .bool(true),
      ]),
      timeout: .seconds(30))
    let data: Data
    let head: HTTPHead
    do {
      (data, head) = try await HTTPTransport.shared.send(request)
    } catch TransportError.redirectRefused {
      throw SignInError(
        .incompatible, "The server redirected the sign-in request. Check the sign-in address.")
    } catch TransportError.offline(timedOut: true) {
      throw SignInError(.unavailable, "The server took too long to respond. Try again.")
    } catch {
      throw SignInError(
        .unavailable, "Can't reach the server. Check the address and your connection.")
    }
    let body = JSONValue.parse(data)
    guard head.isSuccess else { throw failure(head, body) }
    let user = body?["user"]
    guard let token = head.header("set-auth-token"), isSigned(token),
      let userID = user?["id"]?.nonEmptyString,
      let accountEmail = user?["email"]?.nonEmptyString
    else {
      throw SignInError(
        .incompatible,
        "This server doesn't support signing in from the app yet. Ask its administrator to update AutoGPT."
      )
    }
    return SignedInUser(
      sessionToken: token, userID: userID, email: accountEmail, name: user?["name"]?.nonEmptyString)
  }

  /// Exchanges the session credential for a short-lived API access token.
  /// Only the auth server's 401 ends the session; anything else keeps it.
  static func exchange(server: ServerConfig, sessionToken: String) async throws -> AccessGrant {
    var request = HTTPRequest(url: try endpoint(server, "/api/auth/token"))
    request.headers["Authorization"] = "Bearer \(sessionToken)"
    let data: Data
    let head: HTTPHead
    do {
      (data, head) = try await HTTPTransport.shared.send(request)
    } catch TransportError.cancelled {
      throw TransportError.cancelled
    } catch {
      throw AuthError.unavailable(retryAfterSeconds: nil)
    }
    if head.status == 401 { throw AuthError.sessionEnded }
    guard head.isSuccess else {
      throw AuthError.unavailable(retryAfterSeconds: head.retryAfterSeconds)
    }
    guard let token = JSONValue.parse(data)?["token"]?.nonEmptyString else {
      throw AuthError.unavailable(retryAfterSeconds: nil)
    }
    let renewed = head.header("set-auth-token").flatMap {
      isSigned($0) && $0 != sessionToken ? $0 : nil
    }
    return AccessGrant(
      token: token, expiresAt: jwtExpiry(token) ?? Date().addingTimeInterval(5 * 60),
      renewedSessionToken: renewed)
  }

  /// Best-effort server sign-out. The local credential is already gone when this runs.
  static func revoke(server: ServerConfig, sessionToken: String) async {
    guard let url = try? endpoint(server, "/api/auth/sign-out") else { return }
    var request = HTTPRequest(url: url, method: "POST", json: .object([:]), timeout: .seconds(10))
    request.headers["Authorization"] = "Bearer \(sessionToken)"
    _ = try? await HTTPTransport.shared.send(request)
  }

  /// A signed Better Auth bearer credential is `<token>.<signature>`.
  static func isSigned(_ token: String) -> Bool {
    token.contains(".") && !token.hasPrefix(".") && !token.hasSuffix(".")
  }

  private static func endpoint(_ server: ServerConfig, _ path: String) throws -> URL {
    guard let url = URL(string: server.authURL + path) else {
      throw TransportError.offline(timedOut: false)
    }
    return url
  }

  private static func jwtExpiry(_ token: String) -> Date? {
    let parts = token.split(separator: ".")
    guard parts.count >= 2 else { return nil }
    var base64 = parts[1].replacingOccurrences(of: "-", with: "+").replacingOccurrences(
      of: "_", with: "/")
    base64 += String(repeating: "=", count: (4 - base64.count % 4) % 4)
    guard let data = Data(base64Encoded: base64), let exp = JSONValue.parse(data)?["exp"]?.number
    else {
      return nil
    }
    return Date(timeIntervalSince1970: exp)
  }

  private static func failure(_ head: HTTPHead, _ body: JSONValue?) -> SignInError {
    let code = body?["code"]?.string
    if head.status == 401 || code == "INVALID_EMAIL_OR_PASSWORD" {
      return SignInError(.credentials, "That email and password don't match.")
    }
    if code == "EMAIL_NOT_VERIFIED" {
      return SignInError(
        .unverified,
        "Verify your email address first. Check your inbox for a verification link, then sign in again."
      )
    }
    if code == "BANNED_USER" {
      return SignInError(
        .blocked, "This account can't sign in. Contact support if you think this is a mistake.")
    }
    if head.status == 429 {
      let wait = head.retryAfterSeconds
      return SignInError(
        .rateLimited,
        wait.map { "Too many attempts. Try again in \(Int($0.rounded(.up))) seconds." }
          ?? "Too many attempts. Try again shortly.")
    }
    if head.status == 404 {
      return SignInError(.incompatible, "This doesn't look like an AutoGPT sign-in address.")
    }
    if head.status >= 500 {
      return SignInError(.unavailable, "The server couldn't sign you in right now. Try again.")
    }
    let fallback = HTTPError(status: head.status, body: body, retryAfterSeconds: nil)
    return SignInError(.rejected, body?["message"]?.nonEmptyString ?? fallback.message)
  }
}

/// Proves a new session works end to end before it is stored: the auth
/// server must issue an access token for it, and the API must accept that
/// token. A transient API failure doesn't block sign-in; a refusal does.
enum SessionVerifier {
  static func verify(server: ServerConfig, sessionToken: String) async throws -> AccessGrant {
    let grant: AccessGrant
    do {
      grant = try await AuthAPI.exchange(server: server, sessionToken: sessionToken)
    } catch {
      await AuthAPI.revoke(server: server, sessionToken: sessionToken)
      if (error as? AuthError) == .sessionEnded {
        throw SignInError(
          .incompatible, "This server signed you in but didn't accept the app's session.")
      }
      throw SignInError(.unavailable, "The server couldn't finish signing you in. Try again.")
    }
    let probe = BackendClient(server: server, tokens: FixedAccessToken(token: grant.token))
    do {
      _ = try await probe.json("/api/chat/sessions", query: [("limit", "1")])
    } catch {
      let status = (error as? HTTPError)?.status
      let refused = (error as? AuthError) == .sessionRejected || status == 401 || status == 403
      if refused {
        await AuthAPI.revoke(server: server, sessionToken: sessionToken)
        throw SignInError(
          .incompatible,
          "The API address didn't accept this sign-in. Check that both addresses belong to the same AutoGPT server."
        )
      }
    }
    return grant
  }
}
