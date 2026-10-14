import Foundation

protocol AccessTokenProviding: Sendable {
  func accessToken(forceRefresh: Bool) async throws -> String
}

/// Authorized calls to the selected server's API origin. Paths are relative,
/// so an access token can never be attached to another host. A 401 gets one
/// retry with a freshly exchanged token; a second 401 is reported without
/// touching the stored session, which only the auth server can end.
struct BackendClient: Sendable {
  let server: ServerConfig
  let tokens: any AccessTokenProviding
  var transport: HTTPTransport = .shared

  func json(
    _ path: String,
    query: [(String, String?)] = [],
    method: String = "GET",
    body: JSONValue? = nil,
    timeout: Duration = .seconds(20)
  ) async throws -> JSONValue? {
    var request = HTTPRequest(url: try url(path, query: query), method: method)
    request.json = body
    request.timeout = timeout
    let (data, head) = try await authorized(
      request, status: { $0.1.status }, send: { try await transport.send($0) })
    guard head.isSuccess else {
      throw HTTPError(
        status: head.status, body: JSONValue.parse(data),
        retryAfterSeconds: head.retryAfterSeconds)
    }
    if head.status == 204 || data.isEmpty { return nil }
    return JSONValue.parse(data) ?? .string(String(decoding: data, as: UTF8.self))
  }

  func eventStream(
    _ path: String,
    query: [(String, String?)] = [],
    method: String = "GET",
    body: JSONValue? = nil,
    timeout: Duration = .seconds(20)
  ) async throws -> EventStreamResponse {
    var request = HTTPRequest(url: try url(path, query: query), method: method)
    request.json = body
    request.timeout = timeout
    return try await authorized(request, status: \.refusalStatus) {
      try await transport.openEventStream($0)
    }
  }

  private func authorized<Response: Sendable>(
    _ request: HTTPRequest,
    status: (Response) -> Int?,
    send: (HTTPRequest) async throws -> Response
  ) async throws -> Response {
    var first = request
    let token = try await tokens.accessToken(forceRefresh: false)
    first.headers["Authorization"] = "Bearer \(token)"
    let response = try await send(first)
    guard status(response) == 401 else { return response }
    var retry = request
    let fresh = try await tokens.accessToken(forceRefresh: true)
    retry.headers["Authorization"] = "Bearer \(fresh)"
    let second = try await send(retry)
    if status(second) == 401 { throw AuthError.sessionRejected }
    return second
  }

  private func url(_ path: String, query: [(String, String?)]) throws -> URL {
    precondition(path.hasPrefix("/api/"), "Backend paths must be relative API paths")
    let items = query.compactMap { name, value in
      value.map { "\(name.strictlyPercentEncoded)=\($0.strictlyPercentEncoded)" }
    }
    let string = server.apiURL + path + (items.isEmpty ? "" : "?" + items.joined(separator: "&"))
    guard let url = URL(string: string) else { throw TransportError.offline(timedOut: false) }
    return url
  }
}

/// A single token, for proving a new sign-in works before it is stored.
struct FixedAccessToken: AccessTokenProviding {
  let token: String

  func accessToken(forceRefresh: Bool) async throws -> String { token }
}

extension EventStreamResponse {
  var refusalStatus: Int? {
    if case .refused(let head, _) = self { return head.status }
    return nil
  }
}

extension String {
  /// One path segment of an API route.
  var pathSegment: String { strictlyPercentEncoded }
}
