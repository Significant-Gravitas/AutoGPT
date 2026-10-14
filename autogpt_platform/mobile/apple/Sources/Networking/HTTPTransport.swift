import Foundation

struct HTTPHead: Sendable {
  let status: Int
  /// Lowercased header names.
  let headers: [String: String]

  init(_ response: HTTPURLResponse) {
    var headers: [String: String] = [:]
    for (key, value) in response.allHeaderFields {
      if let name = key as? String { headers[name.lowercased()] = "\(value)" }
    }
    self.status = response.statusCode
    self.headers = headers
  }

  var isSuccess: Bool { (200..<300).contains(status) }
  var isRedirect: Bool { (300..<400).contains(status) }

  func header(_ name: String) -> String? { headers[name.lowercased()] }

  var retryAfterSeconds: Double? {
    guard let raw = header("retry-after") ?? header("x-retry-after"),
      let seconds = Double(raw.trimmed), seconds.isFinite, seconds >= 0
    else { return nil }
    return seconds
  }
}

struct HTTPRequest: Sendable {
  var url: URL
  var method = "GET"
  var headers: [String: String] = [:]
  var json: JSONValue?
  /// JSON exchanges: the whole request, body included. Event streams: until
  /// the response headers (or a refusal's bounded body) arrive.
  var timeout: Duration = .seconds(20)

  var urlRequest: URLRequest {
    var request = URLRequest(url: url)
    request.httpMethod = method
    request.httpShouldHandleCookies = false
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
    if let json {
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.httpBody = Data(json.encodedString().utf8)
    }
    return request
  }
}

/// Every request is sent without a cookie jar, cache or credential store, and
/// redirects are refused. A bearer credential may only reach the exact
/// origin it was issued for, and no stored cookie can stand in for it.
final class HTTPTransport: Sendable {
  static let shared = HTTPTransport()

  private let session: URLSession
  private let redirects = RefuseRedirects()

  init() {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.httpCookieAcceptPolicy = .never
    configuration.httpShouldSetCookies = false
    configuration.httpCookieStorage = nil
    configuration.urlCache = nil
    configuration.urlCredentialStorage = nil
    configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    configuration.waitsForConnectivity = false
    // Idle backstop only; deadlines and stream liveness bound each exchange.
    configuration.timeoutIntervalForRequest = 60
    configuration.timeoutIntervalForResource = 35 * 60
    session = URLSession(configuration: configuration, delegate: redirects, delegateQueue: nil)
  }

  /// One whole exchange under a single deadline: connection, headers and body.
  func send(_ request: HTTPRequest) async throws -> (Data, HTTPHead) {
    let urlRequest = request.urlRequest
    let session = self.session
    let redirects = self.redirects
    do {
      let (data, head) = try await withDeadline(request.timeout) {
        let (data, response) = try await session.data(for: urlRequest, delegate: redirects)
        guard let http = response as? HTTPURLResponse else {
          throw TransportError.offline(timedOut: false)
        }
        return (data, HTTPHead(http))
      }
      if head.isRedirect { throw TransportError.redirectRefused }
      return (data, head)
    } catch {
      throw TransportError.from(error)
    }
  }

  func publicImage(_ url: URL) async throws -> Data {
    let session = self.session
    let redirects = self.redirects
    return try await withDeadline(.seconds(15)) {
      var request = URLRequest(url: url)
      request.httpShouldHandleCookies = false
      request.setValue("image/*", forHTTPHeaderField: "Accept")
      let (bytes, response) = try await session.bytes(for: request, delegate: redirects)
      guard let response = response as? HTTPURLResponse, response.statusCode == 200,
        response.mimeType?.hasPrefix("image/") == true,
        response.expectedContentLength <= 4 * 1024 * 1024
      else { throw URLError(.badServerResponse) }
      var data = Data()
      for try await byte in bytes {
        guard data.count < 4 * 1024 * 1024 else { throw URLError(.dataLengthExceedsMaximum) }
        data.append(byte)
      }
      return data
    }
  }

  /// Opens a server-sent event stream. A refusal is returned with its bounded
  /// body rather than thrown, so callers can read its status and details.
  func openEventStream(_ request: HTTPRequest) async throws -> EventStreamResponse {
    var urlRequest = request.urlRequest
    urlRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
    do {
      return try await EventStreamTask().open(
        urlRequest, session: session, headerTimeout: request.timeout)
    } catch {
      throw TransportError.from(error)
    }
  }
}

final class RefuseRedirects: NSObject, URLSessionTaskDelegate, Sendable {
  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest
  ) async -> URLRequest? {
    nil
  }
}
