import Foundation
import Observation

/// The account's conversations, newest first with pinned ones on top, as the server orders them.
@MainActor
@Observable
final class ChatListStore {
  enum Status: Equatable { case idle, loading, ready, error }

  private(set) var sessions: [SessionSummary] = []
  private(set) var total = 0
  private(set) var status: Status = .idle
  private(set) var error: String?
  private(set) var loadingMore = false

  @ObservationIgnored private let api: BackendClient
  @ObservationIgnored private var generation = 0
  private static let page = 30

  init(api: BackendClient) {
    self.api = api
  }

  func refresh() async {
    generation += 1
    let generation = self.generation
    let initial = status != .ready
    if initial { status = .loading }
    error = nil
    do {
      let page = try await ChatAPI.sessions(api, limit: Self.page, offset: 0)
      guard generation == self.generation else { return }
      sessions = page.sessions
      total = page.total
      status = .ready
    } catch {
      guard generation == self.generation else { return }
      status = initial ? .error : .ready
      self.error = ChatErrors.describe(error)
    }
  }

  func loadMore() async {
    guard status == .ready, !loadingMore, sessions.count < total else { return }
    let generation = self.generation
    loadingMore = true
    do {
      let page = try await ChatAPI.sessions(api, limit: Self.page, offset: sessions.count)
      guard generation == self.generation else { return }
      let known = Set(sessions.map(\.id))
      sessions += page.sessions.filter { !known.contains($0.id) }
      total = page.total
    } catch {
      guard generation == self.generation else { return }
      self.error = ChatErrors.describe(error)
    }
    loadingMore = false
  }
}

/// The signed-in account's AI company: Otto is implicit, experts come from the server.
@MainActor
@Observable
final class ExpertDirectory {
  enum Status: Equatable { case idle, loading, ready, error }

  /// Includes let-go experts, so their old conversations still show who they were.
  private(set) var identities: [String: ExpertIdentity] = [:]
  private(set) var roster: [ExpertProfile] = []
  private(set) var status: Status = .idle
  private(set) var error: String?

  @ObservationIgnored private let api: BackendClient
  @ObservationIgnored private var inflight: Task<Void, Never>?

  init(api: BackendClient) {
    self.api = api
  }

  /// Fetches once; later callers reuse what is loaded.
  func ensureLoaded() {
    if status == .idle { Task { await load() } }
  }

  func load() async {
    if let inflight { return await inflight.value }
    let task = Task { await fetch() }
    inflight = task
    await task.value
    inflight = nil
  }

  private func fetch() async {
    if status == .idle || status == .error {
      status = .loading
      error = nil
    }
    let api = self.api
    async let identityResult = capture { try await ExpertsAPI.identities(api) }
    async let rosterResult = capture { try await ExpertsAPI.roster(api) }
    let (identityList, rosterList) = await (identityResult, rosterResult)
    switch (identityList, rosterList) {
    case (.failure(let failure), .failure):
      // A failed refresh keeps whatever is already shown.
      status = roster.isEmpty && identities.isEmpty ? .error : .ready
      error = errorMessage(failure, fallback: "Couldn't load your experts.")
    default:
      let nextRoster = (try? rosterList.get()) ?? roster
      let named = (try? identityList.get()) ?? nextRoster.map(\.identity)
      identities = Dictionary(named.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
      roster = nextRoster
      status = .ready
      if case .failure(let failure) = rosterList {
        error = errorMessage(failure, fallback: "Couldn't load your experts.")
      } else {
        error = nil
      }
    }
  }
}

private func capture<Value: Sendable>(_ operation: @Sendable () async throws -> Value) async
  -> Result<Value, any Error>
{
  do {
    return .success(try await operation())
  } catch {
    return .failure(error)
  }
}

/// Who a conversation is with.
struct Participant: Equatable, Sendable {
  let expertID: String?
  let name: String
  let role: String?
  let avatarURL: URL?
  var avatarAsset: String? = nil
  /// Why the conversation can't take new messages, if it can't.
  let readOnlyReason: String?

  static func otto(webOrigin: String) -> Participant {
    Participant(
      expertID: nil, name: "Otto", role: "Your Head of AI",
      avatarURL: nil, avatarAsset: "OttoPortrait", readOnlyReason: nil)
  }

  static func expert(
    _ identity: ExpertIdentity?, expertID: String, directoryReady: Bool, webOrigin: String
  ) -> Participant {
    guard let identity else {
      // Once the roster is known, an expert missing from it can't be messaged.
      return Participant(
        expertID: expertID, name: directoryReady ? "Former expert" : "Expert", role: nil,
        avatarURL: nil,
        readOnlyReason: directoryReady
          ? "This expert is no longer on your team, so this conversation is read-only." : nil)
    }
    return Participant(
      expertID: expertID, name: identity.name, role: identity.jobTitle ?? identity.role,
      avatarURL: avatarURL(identity.avatarPath, webOrigin: webOrigin),
      readOnlyReason: identity.isArchived
        ? "You let \(identity.name) go, so this conversation is read-only." : nil)
  }

  @MainActor
  static func resolve(_ expertID: String?, directory: ExpertDirectory, webOrigin: String)
    -> Participant
  {
    guard let expertID else { return otto(webOrigin: webOrigin) }
    return expert(
      directory.identities[expertID], expertID: expertID,
      directoryReady: directory.status == .ready,
      webOrigin: webOrigin)
  }

  /// Avatar paths such as `/autogpt-characters/...` are public assets of the
  /// selected server's web app, which also serves its auth routes.
  static func avatarURL(_ path: String?, webOrigin: String) -> URL? {
    guard let path else { return nil }
    if path.hasPrefix("https://") { return URL(string: path) }
    if path.hasPrefix("/"), !path.hasPrefix("//") { return URL(string: webOrigin + path) }
    return nil
  }
}

struct ConversationRoute: Hashable, Identifiable {
  let id = UUID()
  let sessionID: String?
  let expertID: String?

  static func existing(_ sessionID: String) -> ConversationRoute {
    ConversationRoute(sessionID: sessionID, expertID: nil)
  }

  /// A conversation the server creates when its first message is sent.
  static func new(expertID: String?) -> ConversationRoute {
    ConversationRoute(sessionID: nil, expertID: expertID)
  }
}

/// One tab's navigation stack and the conversations open in it. A
/// conversation keeps running while its screen is anywhere in the stack, even
/// when another tab is showing, and is disposed once it is popped.
@MainActor
@Observable
final class ConversationNavigator {
  private(set) var path: [ConversationRoute] = []
  @ObservationIgnored private var controllers: [UUID: ConversationController] = [:]

  func open(_ route: ConversationRoute) {
    path.append(route)
  }

  func setPath(_ next: [ConversationRoute]) {
    path = next
    let live = Set(next.map(\.id))
    for (id, controller) in controllers where !live.contains(id) {
      controller.dispose()
      controllers[id] = nil
    }
  }

  func controller(for route: ConversationRoute, service: any ConversationService)
    -> ConversationController
  {
    if let existing = controllers[route.id] { return existing }
    let controller = ConversationController(
      service: service, sessionID: route.sessionID, expertID: route.expertID)
    controllers[route.id] = controller
    return controller
  }

  func disposeAll() {
    for controller in controllers.values { controller.dispose() }
    controllers.removeAll()
  }

  /// Every live conversation, so scene changes reach those not on screen.
  var openControllers: [ConversationController] { Array(controllers.values) }
}
