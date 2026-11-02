import Foundation

/// A message typed on this device that history doesn't hold yet.
struct LocalMessage: Equatable, Sendable {
  enum State: Sendable { case sending, accepted, failed }

  /// The idempotency key sent with it; a manual retry reuses it.
  let messageID: String
  let text: String
  var state: State
  var error: String?
  let createdAt: Date
}

/// A tool call as the timeline knows it. `output` is nil while no result
/// exists; `state` says whether one is still coming.
struct ToolPart: Equatable, Sendable, Identifiable {
  enum State: Sendable { case running, done, failed, interrupted, unrecorded }

  let toolCallID: String
  let toolName: String
  let displayName: String?
  let input: JSONValue
  let output: JSONValue?
  let state: State

  var id: String { toolCallID }
}

enum AssistantPart: Equatable, Identifiable {
  case text(key: String, text: String, streaming: Bool)
  case reasoning(key: String, text: String, streaming: Bool)
  case tool(key: String, tool: ToolPart)
  case error(key: String, text: String, retryable: Bool)
  case system(key: String, text: String)

  var id: String {
    switch self {
    case .text(let key, _, _), .reasoning(let key, _, _), .tool(let key, _), .error(let key, _, _),
      .system(let key, _):
      key
    }
  }
}

struct UserItem: Equatable {
  let key: String
  let text: String
  let attachments: [MessageAttachment]
  let local: LocalMessage?
}

struct AssistantItem: Equatable {
  let key: String
  var parts: [AssistantPart]
  let live: Bool
}

enum TimelineItem: Equatable, Identifiable {
  case user(UserItem)
  case assistant(AssistantItem)
  /// A post an expert made about its own work, outside an ordinary reply.
  case work(key: String, text: String)

  var id: String {
    switch self {
    case .user(let item): item.key
    case .assistant(let item): item.key
    case .work(let key, _): key
    }
  }
}

enum Timeline {
  /// Persisted rows; built once per history change, not per streamed delta.
  static func historyItems(_ history: [Row], turnRunning: Bool) -> [TimelineItem] {
    var options = Options(
      keyPrefix: "", live: false, outputs: outputs(history),
      unanswered: turnRunning ? .running : .unrecorded, streamingRows: [], skipUserContents: [])
    return items(history, &options)
  }

  /// The running (or just finished) turn, rebuilt as its stream advances.
  static func liveItems(_ live: TurnLog, replayedUserContents: [String]) -> [TimelineItem] {
    let streamingRows = Set(live.blocks.values.compactMap { $0.open ? $0.row : nil })
    var options = Options(
      keyPrefix: "live:", live: true, outputs: outputs(live.rows),
      unanswered: live.status == .running || live.status == .idle ? .running : .interrupted,
      streamingRows: streamingRows, skipUserContents: replayedUserContents)
    return items(live.rows, &options)
  }

  static func localItem(_ local: LocalMessage) -> TimelineItem {
    .user(
      UserItem(key: "local:\(local.messageID)", text: local.text, attachments: [], local: local))
  }

  private struct Options {
    let keyPrefix: String
    let live: Bool
    let outputs: [String: JSONValue]
    let unanswered: ToolPart.State
    let streamingRows: Set<Int>
    var skipUserContents: [String]
  }

  private static func outputs(_ rows: [Row]) -> [String: JSONValue] {
    var outputs: [String: JSONValue] = [:]
    for row in rows where row.role == .tool {
      if let id = row.toolCallID { outputs[id] = JSONValue.parseLoose(row.content) }
    }
    return outputs
  }

  private static func items(_ rows: [Row], _ options: inout Options) -> [TimelineItem] {
    var items: [TimelineItem] = []
    var group: AssistantItem?
    func flush() {
      if let finished = group { items.append(.assistant(finished)) }
      group = nil
    }
    for (index, row) in rows.enumerated() {
      if row.role == .tool { continue }
      if row.role == .user {
        flush()
        if row.metadata?["hidden"]?.bool == true { continue }
        if let skip = options.skipUserContents.firstIndex(of: row.content) {
          options.skipUserContents.remove(at: skip)
          continue
        }
        let split = Attachments.split(row.content)
        items.append(
          .user(
            UserItem(
              key: options.keyPrefix + row.key, text: split.text, attachments: split.attachments,
              local: nil)))
        continue
      }
      if row.metadata?["kind"]?.string == "expert_run" {
        flush()
        if !row.content.trimmed.isEmpty {
          items.append(.work(key: options.keyPrefix + row.key, text: row.content.trimmed))
        }
        continue
      }
      let parts = rowParts(row, streaming: options.streamingRows.contains(index), options)
      if parts.isEmpty { continue }
      if group == nil {
        group = AssistantItem(key: options.keyPrefix + row.key, parts: [], live: options.live)
      }
      group?.parts.append(contentsOf: parts)
    }
    flush()
    return items
  }

  private static func rowParts(_ row: Row, streaming: Bool, _ options: Options) -> [AssistantPart] {
    var parts: [AssistantPart] = []
    let content = row.content
    if row.role == .reasoning {
      if !content.trimmed.isEmpty {
        parts.append(.reasoning(key: row.key, text: content, streaming: streaming))
      }
    } else if content.hasPrefix(Markers.retryableError) {
      parts.append(
        .error(key: row.key, text: markerText(content, Markers.retryableError), retryable: true))
    } else if content.hasPrefix(Markers.error) {
      parts.append(.error(key: row.key, text: markerText(content, Markers.error), retryable: false))
    } else if content.hasPrefix(Markers.system) {
      parts.append(.system(key: row.key, text: markerText(content, Markers.system)))
    } else if !content.trimmed.isEmpty {
      parts.append(.text(key: row.key, text: content, streaming: streaming))
    }
    for call in row.toolCalls where !call.id.isEmpty {
      let output = options.outputs[call.id]
      parts.append(
        .tool(
          key: "\(row.key):\(call.id)",
          tool: ToolPart(
            toolCallID: call.id, toolName: call.name, displayName: call.displayName,
            input: call.input, output: output, state: output != nil ? .done : options.unanswered)))
    }
    return parts
  }

  private static func markerText(_ content: String, _ prefix: String) -> String {
    let text = String(content.dropFirst(prefix.count)).trimmed
    return text.isEmpty ? "Something went wrong." : text
  }
}
