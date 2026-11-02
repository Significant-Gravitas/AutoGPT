import Foundation
import os

/// Short-lived API access tokens for one signed-in account. Tokens live only
/// in memory and concurrent callers share a single exchange. Once disposed,
/// every caller fails immediately, so a request started for a previous
/// account or server can never obtain a token.
actor AccessTokens: AccessTokenProviding {
  private static let refreshMargin: TimeInterval = 60

  private let server: ServerConfig
  private var sessionToken: String
  private var cached: (token: String, expiresAt: Date)?
  private var inflight: Task<String, any Error>?
  private let disposed = OSAllocatedUnfairLock(initialState: false)
  private let onSessionEnded: @Sendable () -> Void
  private let onSessionRenewed: @Sendable (String) -> Void

  init(
    server: ServerConfig,
    sessionToken: String,
    seed: AccessGrant?,
    onSessionEnded: @escaping @Sendable () -> Void,
    onSessionRenewed: @escaping @Sendable (String) -> Void
  ) {
    self.server = server
    self.sessionToken = sessionToken
    self.cached = seed.map { ($0.token, $0.expiresAt) }
    self.onSessionEnded = onSessionEnded
    self.onSessionRenewed = onSessionRenewed
  }

  /// Synchronous, so a sign-out or account switch takes effect before any
  /// later line of code runs.
  nonisolated func dispose() {
    disposed.withLock { $0 = true }
  }

  private var isDisposed: Bool { disposed.withLock { $0 } }

  func accessToken(forceRefresh: Bool) async throws -> String {
    guard !isDisposed else { throw AuthError.sessionEnded }
    if !forceRefresh, let cached, cached.expiresAt.addingTimeInterval(-Self.refreshMargin) > Date()
    {
      return cached.token
    }
    if let inflight { return try await inflight.value }
    let task = Task { try await self.refresh() }
    inflight = task
    defer { inflight = nil }
    return try await task.value
  }

  private func refresh() async throws -> String {
    let token = sessionToken
    do {
      let grant = try await AuthAPI.exchange(server: server, sessionToken: token)
      guard !isDisposed else { throw AuthError.sessionEnded }
      cached = (grant.token, grant.expiresAt)
      if let renewed = grant.renewedSessionToken {
        sessionToken = renewed
        onSessionRenewed(renewed)
      }
      return grant.token
    } catch AuthError.sessionEnded {
      if !isDisposed {
        cached = nil
        dispose()
        onSessionEnded()
      }
      throw AuthError.sessionEnded
    }
  }
}
