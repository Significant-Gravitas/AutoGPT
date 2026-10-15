import Foundation
import Observation

struct SignedInAccount: Equatable, Sendable {
  /// Keychain key of this account's credential; it also partitions caches.
  let key: String
  let userID: String
  let email: String
  let name: String?
}

/// Everything that belongs to one signed-in account on one server. The
/// signed-in UI is rebuilt for every generation, so no screen, cache or
/// request outlives the account it was created for.
struct SignedInContext: Identifiable {
  let generation: Int
  let server: ServerConfig
  let account: SignedInAccount
  let api: BackendClient

  var id: Int { generation }
}

/// Authentication state. Every change of account or server starts a new
/// generation; late results from an older generation are discarded.
@MainActor
@Observable
final class SessionStore {
  enum Phase {
    case restoring
    case signedOut(server: ServerConfig?, notice: String?)
    case signedIn(SignedInContext)
  }

  private(set) var phase: Phase = .restoring
  private(set) var customServers: [ServerConfig] = []
  @ObservationIgnored private(set) var generation = 0
  @ObservationIgnored private var live: LiveSession?
  @ObservationIgnored private var restoreStarted = false

  private struct LiveSession {
    let generation: Int
    let server: ServerConfig
    let account: SignedInAccount
    let tokens: AccessTokens
    var credential: StoredCredential
  }

  var signedOutServer: ServerConfig? {
    if case .signedOut(let server, _) = phase { return server }
    return nil
  }

  var signedOutNotice: String? {
    if case .signedOut(_, let notice) = phase { return notice }
    return nil
  }

  func restore() async {
    guard !restoreStarted else { return }
    restoreStarted = true
    let restored = await Task.detached(priority: .userInitiated) { Restorer.restore() }.value
    switch restored {
    case .failed:
      setSignedOut(server: nil, notice: "Couldn't read your saved sign-in.")
    case .restored(let servers, let server, let signedIn):
      customServers = servers
      if let signedIn {
        activate(
          server: signedIn.server, account: signedIn.account, credential: signedIn.credential,
          seed: nil)
      } else {
        setSignedOut(server: server, notice: nil)
      }
    }
  }

  func chooseServer(_ server: ServerConfig) {
    guard case .signedOut = phase else { return }
    ServerStore.saveSelection(ServerSelection(serverID: server.id, accountKey: nil))
    setSignedOut(server: server, notice: nil)
  }

  func rememberServer(_ server: ServerConfig) {
    customServers = ServerStore.remember(server)
    chooseServer(server)
  }

  func forgetServer(_ serverID: String) {
    customServers = ServerStore.forget(serverID)
    if case .signedOut(let server?, _) = phase, server.id == serverID {
      setSignedOut(server: nil, notice: nil)
    }
  }

  func signIn(email: String, password: String) async throws {
    guard case .signedOut(let server?, _) = phase else {
      throw SignInError(.rejected, "Choose a server first.")
    }
    let generation = self.generation
    let user = try await AuthAPI.signIn(server: server, email: email.trimmed, password: password)
    guard generation == self.generation else {
      throw await abandon(server, user, "The server changed while you were signing in. Try again.")
    }
    let grant = try await SessionVerifier.verify(server: server, sessionToken: user.sessionToken)
    guard generation == self.generation else {
      throw await abandon(server, user, "The server changed while you were signing in. Try again.")
    }
    let accountKey = Digest.stable(user.userID)
    let key = CredentialStore.key(serverPartition: server.partitionKey, accountKey: accountKey)
    let credential = StoredCredential(
      sessionToken: user.sessionToken, authOrigin: server.authURL, apiOrigin: server.apiURL,
      userID: user.userID, email: user.email, name: user.name, savedAt: Date())
    do {
      try CredentialStore.write(key, credential)
      ServerStore.saveSelection(ServerSelection(serverID: server.id, accountKey: accountKey))
    } catch {
      throw await abandon(server, user, "Couldn't save your sign-in on this device. Try again.")
    }
    let account = SignedInAccount(key: key, userID: user.userID, email: user.email, name: user.name)
    activate(server: server, account: account, credential: credential, seed: grant)
  }

  func signOut() async {
    guard case .signedIn = phase, let ended = endLocally(notice: nil) else { return }
    await AuthAPI.revoke(server: ended.server, sessionToken: ended.credential.sessionToken)
  }

  private func abandon(_ server: ServerConfig, _ user: SignedInUser, _ message: String) async
    -> SignInError
  {
    await AuthAPI.revoke(server: server, sessionToken: user.sessionToken)
    return SignInError(.unavailable, message)
  }

  private func activate(
    server: ServerConfig, account: SignedInAccount, credential: StoredCredential, seed: AccessGrant?
  ) {
    live?.tokens.dispose()
    generation += 1
    let generation = self.generation
    let tokens = AccessTokens(
      server: server, sessionToken: credential.sessionToken, seed: seed,
      onSessionEnded: { [weak self] in
        Task { @MainActor in self?.sessionEnded(generation: generation) }
      },
      onSessionRenewed: { [weak self] token in
        Task { @MainActor in self?.renew(generation: generation, sessionToken: token) }
      })
    live = LiveSession(
      generation: generation, server: server, account: account, tokens: tokens,
      credential: credential)
    phase = .signedIn(
      SignedInContext(
        generation: generation, server: server, account: account,
        api: BackendClient(server: server, tokens: tokens)))
  }

  private func sessionEnded(generation: Int) {
    guard live?.generation == generation else { return }
    endLocally(notice: "Your session ended. Sign in again.")
  }

  private func renew(generation: Int, sessionToken: String) {
    guard var current = live, current.generation == generation else { return }
    current.credential.sessionToken = sessionToken
    current.credential.savedAt = Date()
    live = current
    try? CredentialStore.write(current.account.key, current.credential)
  }

  /// The local credential goes first, so nothing can reuse it whatever happens next.
  @discardableResult
  private func endLocally(notice: String?) -> LiveSession? {
    guard let current = live else { return nil }
    live = nil
    current.tokens.dispose()
    CredentialStore.delete(current.account.key)
    ServerStore.saveSelection(ServerSelection(serverID: current.server.id, accountKey: nil))
    setSignedOut(server: current.server, notice: notice)
    return current
  }

  private func setSignedOut(server: ServerConfig?, notice: String?) {
    generation += 1
    phase = .signedOut(server: server, notice: notice)
  }
}

private enum Restorer {
  struct SignedIn: Sendable {
    let server: ServerConfig
    let account: SignedInAccount
    let credential: StoredCredential
  }

  enum Result: Sendable {
    case failed
    case restored(customServers: [ServerConfig], server: ServerConfig?, signedIn: SignedIn?)
  }

  static func restore() -> Result {
    CredentialStore.wipeLeftoversFromPreviousInstall()
    let servers = ServerStore.customServers()
    let selection = ServerStore.selection()
    let server = selection.flatMap { ServerStore.server(id: $0.serverID) }
    guard let server, let accountKey = selection?.accountKey else {
      return .restored(customServers: servers, server: server, signedIn: nil)
    }
    let key = CredentialStore.key(serverPartition: server.partitionKey, accountKey: accountKey)
    let credential: StoredCredential?
    do {
      credential = try CredentialStore.read(key)
    } catch {
      return .failed
    }
    // A credential is only used with the exact origins it was issued for.
    guard let credential, credential.authOrigin == server.authURL,
      credential.apiOrigin == server.apiURL,
      Digest.stable(credential.userID) == accountKey
    else {
      ServerStore.saveSelection(ServerSelection(serverID: server.id, accountKey: nil))
      return .restored(customServers: servers, server: server, signedIn: nil)
    }
    let account = SignedInAccount(
      key: key, userID: credential.userID, email: credential.email, name: credential.name)
    return .restored(
      customServers: servers, server: server,
      signedIn: SignedIn(server: server, account: account, credential: credential))
  }
}
