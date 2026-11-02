import Foundation

struct StreamEntry: Sendable {
  let turn: String
  let entryID: String
  let chunk: WireChunk
}

/// The client's model of one running turn: the rows the backend persists for
/// it, plus what rows cannot hold yet (open blocks, pending tools). The row
/// rules mirror the web client's `stream/turnLog.ts`, which in turn mirrors
/// the backend's `_dispatch_response` and pending-message persistence.
struct TurnLog: Equatable, Sendable {
  enum Status: Equatable, Sendable { case idle, running, finished, failed }

  struct Block: Equatable, Sendable {
    enum Kind: Sendable { case text, reasoning }
    var kind: Kind
    var open: Bool
    var row: Int?
  }

  struct ToolState: Equatable, Sendable {
    enum Phase: Sendable { case inputStreaming, inputAvailable, outputAvailable }
    var name: String
    var phase: Phase
  }

  var turnID: String?
  /// The last applied entry id; nil until one is applied (a cursor of `0-0`).
  var cursor: String?
  var status: Status = .idle
  var messageID: String?
  var rows: [Row] = []
  var blocks: [String: Block] = [:]
  var tools: [String: ToolState] = [:]
  var displayNames: [String: String] = [:]
  var assistantRow: Int?
  var hasToolResults = false
  var providerFailure: [String: JSONValue]?
  /// The latest `data-status` text, shown while nothing else is streaming.
  var statusText: String?
  var protocolErrors = 0

  private static let retryableCodes: Set<String> = [
    "transient_api_error", "empty_completion", "baseline_error",
  ]

  // MARK: Row rules

  /// Rule 1: a text delta extends the assistant row, or opens one after tool results.
  mutating func appendText(blockID: String, delta: String) {
    if hasToolResults, assistantRow != nil {
      rows.append(.assistant("assistant:\(blockID)", content: delta))
      assistantRow = rows.count - 1
      hasToolResults = false
      return
    }
    guard let index = assistantRow else {
      rows.append(.assistant("assistant:\(blockID)", content: delta))
      assistantRow = rows.count - 1
      return
    }
    rows[index].content += delta
  }

  /// Rule 2: every reasoning block is its own row, appended at its start.
  mutating func openReasoning(blockID: String) {
    var row = Row.assistant("reasoning:\(blockID)", content: "")
    row.role = .reasoning
    rows.append(row)
  }

  /// Rule 3: a reasoning delta extends its block's row.
  mutating func appendReasoning(row: Int, delta: String) {
    guard rows.indices.contains(row) else { return }
    rows[row].content += delta
  }

  /// Rule 4: a tool call joins the current assistant row, or opens one after tool results.
  mutating func addToolCall(id: String, name: String, input: JSONValue) {
    let call = ToolCall(id: id, name: name, input: input, displayName: displayNames[id])
    if let index = assistantRow, !hasToolResults {
      rows[index].toolCalls.append(call)
      return
    }
    var row = Row.assistant("assistant:\(id)", content: "")
    row.toolCalls = [call]
    rows.append(row)
    assistantRow = rows.count - 1
    hasToolResults = false
  }

  /// Rule 5: a tool result is its own row, once per call.
  mutating func addToolResult(toolCallID: String, output: JSONValue?) {
    if rows.contains(where: { $0.role == .tool && $0.toolCallID == toolCallID }) { return }
    let content: String
    switch output {
    case .string(let text)?: content = text
    case let value?: content = value.encodedString()
    case nil: content = ""
    }
    var row = Row.assistant("tool:\(toolCallID)", content: content)
    row.role = .tool
    row.toolCallID = toolCallID
    rows.append(row)
    hasToolResults = true
  }

  /// Rule 6: an error becomes a marker row, unless one already ends the turn.
  mutating func appendErrorMarker(entryID: String, errorText: String) {
    let failure = providerFailure
    providerFailure = nil
    status = .failed
    if let last = rows.last, last.isMarker { return }
    let (code, remainder) = Self.splitCodePrefix(errorText)
    let display = failure?["message"]?.string ?? remainder
    let retryable =
      failure.map { $0["retryable"]?.bool == true }
      ?? code.map { Self.retryableCodes.contains($0) } ?? false
    let prefix = retryable ? Markers.retryableError : Markers.error
    var row = Row.assistant("error:\(entryID)", content: "\(prefix) \(display)")
    row.metadata = failure.map { ["provider_failure": .object($0)] }
    rows.append(row)
  }

  /// Rule 7: a display name lands on its tool call, now or when the call arrives.
  mutating func setToolDisplayName(toolCallID: String, name: String) {
    displayNames[toolCallID] = name
    guard let index = rows.firstIndex(where: { $0.toolCalls.contains { $0.id == toolCallID } })
    else { return }
    for call in rows[index].toolCalls.indices where rows[index].toolCalls[call].id == toolCallID {
      rows[index].toolCalls[call].displayName = name
    }
  }

  /// Rule 8: drained follow-ups are user rows, keyed by pending id, where the hint arrived.
  mutating func appendDrainedUserRows(_ messages: [(id: String, content: String)]) {
    var known = Set(rows.map(\.key))
    for message in messages {
      var row = Row.assistant("user:\(message.id)", content: message.content)
      row.role = .user
      guard !known.contains(row.key) else { continue }
      known.insert(row.key)
      rows.append(row)
    }
  }

  /// The inverse of the row rules at a checkpoint: the turn's persisted rows
  /// seed a log that the stream after the checkpoint continues. At a
  /// checkpoint no block is open and no tool call is pending.
  static func seeded(turnID: String, rows: [Row], checkpoint: TurnCheckpoint?) -> TurnLog {
    var log = TurnLog(turnID: turnID, status: .running)
    guard let checkpoint else { return log }
    let seeded =
      rows
      .filter {
        guard let sequence = $0.sequence else { return false }
        return sequence >= checkpoint.sequence && sequence < checkpoint.sequence + checkpoint.rows
      }
      .sorted { ($0.sequence ?? 0) < ($1.sequence ?? 0) }
    let lastAssistant = seeded.lastIndex { $0.role == .assistant && !$0.isMarker }
    let answered = Set(seeded.compactMap(\.toolCallID))
    for call in seeded.flatMap(\.toolCalls) {
      log.tools[call.id] = ToolState(
        name: call.name, phase: answered.contains(call.id) ? .outputAvailable : .inputAvailable)
    }
    log.cursor = checkpoint.entryID
    log.rows = seeded
    log.assistantRow = lastAssistant
    log.hasToolResults =
      lastAssistant.map { seeded[($0 + 1)...].contains { $0.role == .tool } } ?? false
    return log
  }

  /// `[code:<id>] ` at the start of an error names its cause.
  private static func splitCodePrefix(_ text: String) -> (code: String?, remainder: String) {
    let leading = text.drop { $0.isWhitespace }
    guard leading.hasPrefix("[code:"), let close = leading.firstIndex(of: "]") else {
      return (nil, text)
    }
    let code = String(leading[leading.index(leading.startIndex, offsetBy: 6)..<close])
    var rest = leading[leading.index(after: close)...]
    if rest.hasPrefix(" ") { rest = rest.dropFirst() }
    return (code, String(rest))
  }
}

// MARK: Entry application

extension TurnLog {
  /// Applies one stored entry. Idempotent over any suffix: an entry at or
  /// before the cursor is dropped before any rule runs, so overlapping
  /// deliveries after a reconnect converge. Unknown chunk types are counted
  /// and ignored; they never stop the turn from rendering.
  mutating func apply(_ entry: StreamEntry) {
    if let turnID, entry.turn != turnID {
      protocolErrors += 1
      return
    }
    if let cursor, EntryID.compare(entry.entryID, cursor) <= 0 { return }
    turnID = entry.turn
    cursor = entry.entryID
    if status == .idle { status = .running }
    applyChunk(entryID: entry.entryID, entry.chunk)
  }

  private mutating func applyChunk(entryID: String, _ chunk: WireChunk) {
    switch chunk.type {
    case "start":
      // An auto-continue streams a second message into the same turn.
      guard let id = chunk["messageId"]?.string, !id.isEmpty, id != messageID else { return }
      messageID = id
      assistantRow = nil
      hasToolResults = false
    case "finish":
      status = status == .failed ? .failed : .finished
    case "start-step":
      break
    case "finish-step":
      for (id, block) in blocks where block.open { blocks[id]?.open = false }
    case "text-start", "reasoning-start":
      startBlock(chunk)
    case "text-delta", "reasoning-delta":
      extendBlock(chunk)
    case "text-end", "reasoning-end":
      let id = Self.text(chunk["id"])
      if blocks[id]?.open == true { blocks[id]?.open = false }
    case "tool-input-start":
      let id = Self.text(chunk["toolCallId"])
      guard !id.isEmpty, tools[id] == nil else { return }
      tools[id] = ToolState(name: Self.text(chunk["toolName"]), phase: .inputStreaming)
    case "tool-input-available":
      receiveToolInput(chunk)
    case "tool-output-available":
      receiveToolOutput(chunk)
    case "error":
      appendErrorMarker(entryID: entryID, errorText: Self.text(chunk["errorText"]))
    case "data-tool-display":
      let data = chunk["data"]
      setToolDisplayName(
        toolCallID: Self.text(data?["toolCallId"]), name: Self.text(data?["displayName"]))
    case "data-pending-drained":
      let messages = chunk["data"]?["messages"]?.array ?? []
      appendDrainedUserRows(
        messages.map { (id: Self.text($0["id"]), content: Self.text($0["content"])) })
    case "data-status":
      if let message = chunk["data"]?["message"]?.string { statusText = message }
    case "data-provider-failure":
      providerFailure = chunk["data"]?.object
    default:
      // Other data parts (checkpoints, compaction progress, mode changes,
      // dream operations) are transient and carry nothing this view draws.
      if !chunk.type.hasPrefix("data-") { protocolErrors += 1 }
    }
  }

  private mutating func startBlock(_ chunk: WireChunk) {
    let id = Self.text(chunk["id"])
    guard blocks[id] == nil else { return }
    if chunk.type == "text-start" {
      blocks[id] = Block(kind: .text, open: true, row: nil)
    } else {
      let row = rows.count
      openReasoning(blockID: id)
      blocks[id] = Block(kind: .reasoning, open: true, row: row)
    }
  }

  private mutating func extendBlock(_ chunk: WireChunk) {
    let id = Self.text(chunk["id"])
    let kind: Block.Kind = chunk.type == "text-delta" ? .text : .reasoning
    guard let block = blocks[id], block.kind == kind, block.open else {
      protocolErrors += 1
      return
    }
    let delta = Self.text(chunk["delta"])
    if kind == .reasoning, let row = block.row {
      appendReasoning(row: row, delta: delta)
      return
    }
    appendText(blockID: id, delta: delta)
    blocks[id]?.row = assistantRow
  }

  private mutating func receiveToolInput(_ chunk: WireChunk) {
    let id = Self.text(chunk["toolCallId"])
    guard !id.isEmpty else {
      protocolErrors += 1
      return
    }
    let known = tools[id]
    if let known, known.phase != .inputStreaming { return }
    let named = Self.text(chunk["toolName"])
    let name = named.isEmpty ? known?.name ?? "" : named
    addToolCall(id: id, name: name, input: chunk["input"].present ?? .object([:]))
    tools[id] = ToolState(name: name, phase: .inputAvailable)
  }

  // An output for a call this log never saw is still a row the backend persists.
  private mutating func receiveToolOutput(_ chunk: WireChunk) {
    let id = Self.text(chunk["toolCallId"])
    guard !id.isEmpty else {
      protocolErrors += 1
      return
    }
    let known = tools[id]
    if known?.phase == .outputAvailable { return }
    addToolResult(toolCallID: id, output: chunk["output"])
    tools[id] = ToolState(name: known?.name ?? "", phase: .outputAvailable)
  }

  private static func text(_ value: JSONValue?) -> String { value?.string ?? "" }
}

enum EntryID {
  /// Redis stream ids are `<ms>-<seq>`, ordered numerically on both halves.
  static func compare(_ a: String, _ b: String) -> Int {
    let left = parts(a)
    let right = parts(b)
    if left.ms != right.ms { return left.ms < right.ms ? -1 : 1 }
    if left.seq != right.seq { return left.seq < right.seq ? -1 : 1 }
    return 0
  }

  private static func parts(_ id: String) -> (ms: UInt64, seq: UInt64) {
    let pieces = id.split(separator: "-", maxSplits: 1)
    return (
      pieces.first.flatMap { UInt64($0) } ?? 0,
      pieces.count > 1 ? UInt64(pieces[1]) ?? 0 : 0
    )
  }
}
