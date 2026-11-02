import Foundation

struct SessionSummary: Identifiable, Equatable, Sendable {
  let id: String
  let createdAt: String?
  let updatedAt: String?
  let title: String?
  let chatStatus: String
  let isProcessing: Bool
  let isPinned: Bool
  let expertID: String?

  init?(_ raw: JSONValue) {
    guard let id = raw["id"]?.nonEmptyString else { return nil }
    self.id = id
    createdAt = raw["created_at"]?.string
    updatedAt = raw["updated_at"]?.string
    title = raw["title"]?.nonEmptyString
    chatStatus = raw["chat_status"]?.string ?? "idle"
    isProcessing = raw["is_processing"]?.bool == true
    isPinned = raw["is_pinned"]?.bool == true
    expertID = raw["expert_id"]?.nonEmptyString
  }
}

struct SessionPage: Sendable {
  let sessions: [SessionSummary]
  let total: Int
}

/// Where a running turn's last checkpoint sits: its first `rows` rows start at DB `sequence`.
struct TurnCheckpoint: Equatable, Sendable {
  let entryID: String
  let rows: Int
  let sequence: Int

  init(entryID: String, rows: Int, sequence: Int) {
    self.entryID = entryID
    self.rows = rows
    self.sequence = sequence
  }

  init?(_ raw: JSONValue?) {
    guard let entryID = raw?["entry_id"]?.nonEmptyString, let rows = raw?["rows"]?.int,
      let sequence = raw?["sequence"]?.int
    else { return nil }
    self.init(entryID: entryID, rows: rows, sequence: sequence)
  }
}

struct ActiveStream: Equatable, Sendable {
  let turnID: String
  let startedAt: String?
  let checkpoint: TurnCheckpoint?
}

struct SessionDetail: Sendable {
  let id: String
  let chatStatus: String
  let messages: [JSONValue]
  let activeStream: ActiveStream?
  let hasMore: Bool
  let oldestSequence: Int?
  let expertID: String?

  init(
    id: String, chatStatus: String, messages: [JSONValue], activeStream: ActiveStream?,
    hasMore: Bool, oldestSequence: Int?, expertID: String?
  ) {
    self.id = id
    self.chatStatus = chatStatus
    self.messages = messages
    self.activeStream = activeStream
    self.hasMore = hasMore
    self.oldestSequence = oldestSequence
    self.expertID = expertID
  }

  init(_ body: JSONValue?, fallbackID: String) {
    let active = body?["active_stream"]
    let turnID = active?["turn_id"]?.nonEmptyString
    self.init(
      id: body?["id"]?.string ?? fallbackID,
      chatStatus: body?["chat_status"]?.string ?? "idle",
      messages: body?["messages"]?.array ?? [],
      activeStream: turnID.map {
        ActiveStream(
          turnID: $0, startedAt: active?["started_at"]?.string,
          checkpoint: TurnCheckpoint(active?["checkpoint"]))
      },
      hasMore: body?["has_more_messages"]?.bool == true,
      oldestSequence: body?["oldest_sequence"]?.int,
      expertID: body?["expert_id"]?.nonEmptyString)
  }
}

struct CreatedSession: Sendable {
  let id: String
  let expertID: String?
}

enum TurnOpen: Sendable {
  case accepted(AsyncThrowingStream<SSEFrame, any Error>)
  case refused(HTTPError)
}

enum ChatAPI {
  static func sessions(
    _ api: BackendClient, limit: Int, offset: Int, expertID: String? = nil, pinnedFirst: Bool? = nil
  ) async throws -> SessionPage {
    let body = try await api.json(
      "/api/chat/sessions",
      query: [
        ("limit", String(limit)), ("offset", String(offset)), ("expert_id", expertID),
        ("pinned_first", pinnedFirst.map { $0 ? "true" : "false" }),
      ])
    let raw = body?["sessions"]?.array ?? []
    return SessionPage(
      sessions: raw.compactMap(SessionSummary.init), total: body?["total"]?.int ?? raw.count)
  }

  static func detail(_ api: BackendClient, sessionID: String, limit: Int, beforeSequence: Int?)
    async throws -> SessionDetail
  {
    let body = try await api.json(
      "/api/chat/sessions/\(sessionID.pathSegment)",
      query: [("limit", String(limit)), ("before_sequence", beforeSequence.map(String.init))])
    return SessionDetail(body, fallbackID: sessionID)
  }

  static func create(_ api: BackendClient, expertID: String?) async throws -> CreatedSession {
    let body = try await api.json(
      "/api/chat/sessions", method: "POST",
      body: .object(expertID.map { ["expert_id": .string($0)] } ?? [:]))
    guard let id = body?["id"]?.nonEmptyString else {
      throw HTTPError(
        status: 502, body: .string("The server didn't return a conversation."),
        retryAfterSeconds: nil)
    }
    return CreatedSession(id: id, expertID: body?["expert_id"]?.nonEmptyString)
  }

  /// Starts a turn. `messageID` is the per-send idempotency key; retries reuse it.
  static func openTurn(_ api: BackendClient, sessionID: String, message: String, messageID: String)
    async throws -> TurnOpen
  {
    try turnOpen(
      await api.eventStream(
        "/api/chat/sessions/\(sessionID.pathSegment)/stream", method: "POST",
        body: .object([
          "message": .string(message), "is_user_message": .bool(true),
          "message_id": .string(messageID),
        ]),
        timeout: .seconds(30)))
  }

  /// Reads a turn's stored entries after `after`, then follows it while it runs.
  static func resumeTurn(_ api: BackendClient, sessionID: String, turnID: String, after: String)
    async throws -> TurnOpen
  {
    try turnOpen(
      await api.eventStream(
        "/api/chat/sessions/\(sessionID.pathSegment)/stream",
        query: [("turn", turnID), ("after", after)]))
  }

  static func cancelTurn(_ api: BackendClient, sessionID: String) async throws -> Bool {
    let body = try await api.json(
      "/api/chat/sessions/\(sessionID.pathSegment)/cancel", method: "POST", timeout: .seconds(15))
    return body?["cancelled"]?.bool == true
  }

  private static func turnOpen(_ response: EventStreamResponse) throws -> TurnOpen {
    switch response {
    case .open(_, let frames):
      return .accepted(frames)
    case .refused(let head, let body):
      return .refused(
        HTTPError(
          status: head.status, body: JSONValue.parse(body),
          retryAfterSeconds: head.retryAfterSeconds))
    }
  }
}

enum ChatErrors {
  /// User-facing copy for the chat routes' documented refusals.
  static func describe(_ error: any Error) -> String {
    guard let error = error as? HTTPError else {
      return errorMessage(error, fallback: "Something went wrong. Try again.")
    }
    switch error.code {
    case "chat_transport_selection_required":
      return "Choose which AI connection new chats use in AutoGPT on the web, then try again."
    case "chat_transport_not_configured":
      return "This server has no AI connection set up for chat."
    case "advanced_tier_unavailable", "advanced_tier_not_entitled":
      return "That model tier isn't available on this account."
    default:
      break
    }
    switch error.status {
    case 402:
      return
        "A subscription is needed to chat on this server. Subscribing in the app isn't available yet."
    case 404:
      return
        "This conversation isn't available. It may have been deleted, or the expert was let go."
    case 429:
      return error.message.hasPrefix("Request failed")
        ? "You've reached a usage limit. Try again later." : error.message
    case 503:
      return error.retryAfterSeconds.map {
        "The chat service is busy. Try again in \(Int($0.rounded(.up))) seconds."
      } ?? "The chat service is busy. Try again in a moment."
    default:
      return error.message
    }
  }
}
