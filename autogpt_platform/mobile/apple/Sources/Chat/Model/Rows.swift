import Foundation

struct ToolCall: Equatable, Sendable {
  let id: String
  let name: String
  let input: JSONValue
  var displayName: String?
}

/// One conversation row, shaped like the backend's persisted ChatMessage.
/// History pages and a live turn both produce rows, so a reload renders
/// exactly what the stream showed.
struct Row: Equatable, Sendable {
  enum Role: String, Sendable { case user, assistant, reasoning, tool }

  let key: String
  var role: Role
  var content: String
  var toolCalls: [ToolCall] = []
  var toolCallID: String?
  var sequence: Int?
  var metadata: [String: JSONValue]?
  var createdAt: String?

  static func assistant(_ key: String, content: String) -> Row {
    Row(key: key, role: .assistant, content: content)
  }

  var isMarker: Bool {
    role == .assistant && Markers.all.contains { content.hasPrefix($0) }
  }
}

/// Marker prefixes, kept in step with backend `copilot/constants.py`.
enum Markers {
  static let error = "[__COPILOT_ERROR_f7a1__]"
  static let retryableError = "[__COPILOT_RETRYABLE_ERROR_a9c2__]"
  static let system = "[__COPILOT_SYSTEM_e3b0__]"
  static let all = [error, retryableError, system]
}

enum Rows {
  /// Rows from `GET /sessions/{id}` messages; unknown roles (e.g. system) are skipped.
  static func fromPersisted(_ messages: [JSONValue]) -> [Row] {
    messages.enumerated().compactMap { index, raw in
      guard let record = raw.object, let role = record["role"]?.string.flatMap(Row.Role.init) else {
        return nil
      }
      let sequence = record["sequence"]?.int
      return Row(
        key: sequence.map { "seq:\($0)" } ?? "idx:\(index):\(record["id"]?.string ?? "")",
        role: role,
        content: record["content"]?.string ?? "",
        toolCalls: (record["tool_calls"]?.array ?? []).map(toolCall),
        toolCallID: record["tool_call_id"]?.string,
        sequence: sequence,
        metadata: record["metadata"]?.object,
        createdAt: record["created_at"]?.string)
    }
  }

  /// Union of loaded pages, ordered by sequence; a later copy of a row wins.
  static func merge(older: [Row], newer: [Row]) -> [Row] {
    var bySequence: [Int: Row] = [:]
    var unsequenced: [Row] = []
    for row in older + newer {
      if let sequence = row.sequence { bySequence[sequence] = row } else { unsequenced.append(row) }
    }
    return bySequence.keys.sorted().compactMap { bySequence[$0] } + unsequenced
  }

  private static func toolCall(_ raw: JSONValue) -> ToolCall {
    let function = raw["function"]
    let arguments = function?["arguments"]
    let input: JSONValue
    if let text = arguments?.string {
      input = text.trimmed.isEmpty ? .object([:]) : JSONValue.parseLoose(text)
    } else {
      input = arguments.present ?? .object([:])
    }
    return ToolCall(
      id: scalarString(raw["id"]), name: scalarString(function?["name"]), input: input,
      displayName: raw["display_name"]?.string)
  }

  private static func scalarString(_ value: JSONValue?) -> String {
    switch value {
    case .string(let text)?: text
    case .number(let number)?: number.rounded() == number ? String(Int(number)) : String(number)
    case .bool(let flag)?: String(flag)
    default: ""
    }
  }
}
