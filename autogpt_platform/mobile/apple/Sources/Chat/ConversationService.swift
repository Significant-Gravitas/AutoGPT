import Foundation

/// What a conversation needs from a server. The live implementation is the
/// production chat API; Debug builds add a UI-protocol fixture that never
/// leaves the device.
protocol ConversationService: Sendable {
  func detail(sessionID: String, limit: Int, beforeSequence: Int?) async throws -> SessionDetail
  func create(expertID: String?) async throws -> CreatedSession
  func openTurn(sessionID: String, message: String, messageID: String) async throws -> TurnOpen
  func resumeTurn(sessionID: String, turnID: String, after: String) async throws -> TurnOpen
  func cancelTurn(sessionID: String) async throws -> Bool
}

struct LiveConversationService: ConversationService {
  let api: BackendClient

  func detail(sessionID: String, limit: Int, beforeSequence: Int?) async throws -> SessionDetail {
    try await ChatAPI.detail(
      api, sessionID: sessionID, limit: limit, beforeSequence: beforeSequence)
  }

  func create(expertID: String?) async throws -> CreatedSession {
    try await ChatAPI.create(api, expertID: expertID)
  }

  func openTurn(sessionID: String, message: String, messageID: String) async throws -> TurnOpen {
    try await ChatAPI.openTurn(api, sessionID: sessionID, message: message, messageID: messageID)
  }

  func resumeTurn(sessionID: String, turnID: String, after: String) async throws -> TurnOpen {
    try await ChatAPI.resumeTurn(api, sessionID: sessionID, turnID: turnID, after: after)
  }

  func cancelTurn(sessionID: String) async throws -> Bool {
    try await ChatAPI.cancelTurn(api, sessionID: sessionID)
  }
}
