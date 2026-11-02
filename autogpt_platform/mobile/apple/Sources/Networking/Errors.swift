import Foundation

/// An error whose text is safe and meaningful to show. None of these ever
/// carries a credential.
protocol UserFacingError: Error {
  var userMessage: String { get }
}

/// The request never produced a usable response.
enum TransportError: UserFacingError, Equatable {
  case offline(timedOut: Bool)
  /// Credentials are never sent across a redirect, so a redirect is a failure.
  case redirectRefused
  case cancelled

  var userMessage: String {
    switch self {
    case .offline(true): "The server took too long to respond."
    case .offline(false): "Can't reach the server."
    case .redirectRefused: "The server redirected the request somewhere else, so it wasn't sent."
    case .cancelled: "The request was cancelled."
    }
  }

  static func from(_ error: any Error) -> any Error {
    if error is TransportError || error is HTTPError { return error }
    if error is CancellationError { return TransportError.cancelled }
    if error is DeadlineExceeded { return TransportError.offline(timedOut: true) }
    if let url = error as? URLError {
      switch url.code {
      case .cancelled:
        return Task.isCancelled ? TransportError.cancelled : TransportError.offline(timedOut: false)
      case .timedOut: return TransportError.offline(timedOut: true)
      default: return TransportError.offline(timedOut: false)
      }
    }
    return TransportError.offline(timedOut: false)
  }
}

/// The server answered with a non-success status.
struct HTTPError: UserFacingError {
  let status: Int
  let code: String?
  let message: String
  /// The whole parsed body, for refusals that carry data such as a checkpoint.
  let body: JSONValue?
  let retryAfterSeconds: Double?

  init(status: Int, body: JSONValue?, retryAfterSeconds: Double?) {
    let described = Self.describe(body)
    self.status = status
    self.code = described.code
    self.message = described.message ?? "Request failed (\(status))"
    self.body = body
    self.retryAfterSeconds = retryAfterSeconds
  }

  var userMessage: String { message }

  /// FastAPI puts a string or an object in `detail`; Better Auth uses
  /// `{code, message}`. Validation arrays become a generic sentence.
  private static func describe(_ body: JSONValue?) -> (message: String?, code: String?) {
    guard let body else { return (nil, nil) }
    if body.string != nil { return (body.nonEmptyString, nil) }
    guard let record = body.object else { return (nil, nil) }
    let detail = record["detail"] ?? body
    if let text = detail.nonEmptyString {
      let isCode = text.range(of: #"^[a-z0-9_]+$"#, options: .regularExpression) != nil
      return (text, isCode ? text : nil)
    }
    if detail.array != nil {
      return ("The server couldn't accept that request.", "validation_error")
    }
    guard detail.object != nil else { return (nil, nil) }
    return (detail["message"]?.nonEmptyString, detail["code"]?.string ?? detail["kind"]?.string)
  }
}

enum AuthError: UserFacingError, Equatable {
  /// The auth server says this session no longer exists: expired, revoked or signed out.
  case sessionEnded
  /// The API refused a freshly issued access token. The stored session is left intact.
  case sessionRejected
  /// The auth service couldn't decide right now; the session must be kept.
  case unavailable(retryAfterSeconds: Double?)

  var userMessage: String {
    switch self {
    case .sessionEnded: "Your session ended. Sign in again."
    case .sessionRejected:
      "This server didn't accept your session. Try again, or sign out and sign back in."
    case .unavailable: "Sign-in service is temporarily unavailable. Try again shortly."
    }
  }
}

func errorMessage(_ error: any Error, fallback: String) -> String {
  (error as? any UserFacingError)?.userMessage ?? fallback
}
