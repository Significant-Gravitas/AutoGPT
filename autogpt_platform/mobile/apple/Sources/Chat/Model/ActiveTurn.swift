import Foundation

/// Which persisted rows a resumed stream draws itself.
enum TurnMembership: Equatable, Sendable {
  case fromSequence(Int)
  case createdSince(Date?)

  func contains(_ row: Row) -> Bool {
    switch self {
    case .fromSequence(let start):
      return row.sequence.map { $0 >= start } ?? false
    case .createdSince(let start):
      guard let start, row.role != .user else { return false }
      return Self.created(row, since: start)
    }
  }

  static func created(_ row: Row, since start: Date) -> Bool {
    guard let created = row.createdAt.flatMap(ISODate.parse) else { return false }
    return created >= start
  }
}

/// Divides loaded history from a turn that is still running, so resuming it
/// neither drops nor duplicates rows.
///
/// With a checkpoint, the turn's first `rows` rows from `sequence` are already
/// persisted: they seed the turn and the stream resumes after the checkpoint
/// entry. Rows persisted after it come back through the stream.
///
/// Without one the whole turn is replayed from the start. Assistant-side rows
/// created since the turn started belong to it. The prompt that started the
/// turn stays in history (it is persisted before the turn registers, and the
/// stream never repeats it); follow-ups drained into the turn are replayed
/// and are de-duplicated by content instead.
struct ActiveTurnSplit {
  let membership: TurnMembership
  let seed: TurnLog
  /// The stream entry to resume after.
  let after: String
  let replayedUserContents: [String]

  init(rows: [Row], active: ActiveStream) {
    if let checkpoint = active.checkpoint {
      membership = .fromSequence(checkpoint.sequence)
      seed = .seeded(turnID: active.turnID, rows: rows, checkpoint: checkpoint)
      after = checkpoint.entryID
      replayedUserContents = []
      return
    }
    let startedAt = active.startedAt.flatMap(ISODate.parse)
    membership = .createdSince(startedAt)
    seed = .seeded(turnID: active.turnID, rows: [], checkpoint: nil)
    after = "0-0"
    replayedUserContents =
      startedAt.map { start in
        rows.filter { $0.role == .user && TurnMembership.created($0, since: start) }.map(\.content)
      } ?? []
  }
}

/// The backend appends an `[Attached files]` block (`build_files_block`) to a
/// user message that carries files or folders. It is shown as attachment
/// chips rather than raw text.
struct MessageAttachment: Equatable, Sendable, Identifiable {
  enum Kind: Sendable { case file, folder }
  let kind: Kind
  let id: String
  let name: String
  let detail: String
}

enum Attachments {
  private static let fileHint = "Use read_workspace_file with the file_id to access file contents."
  private static let folderHint =
    "Use list_workspace_files with the folder_id to see what is in a folder."
  private static let block: NSRegularExpression = {
    let hints =
      "(?:\(NSRegularExpression.escapedPattern(for: fileHint))|\(NSRegularExpression.escapedPattern(for: folderHint)))"
    return try! NSRegularExpression(
      pattern: #"\n?\n?\[Attached files\]\n([\s\S]*?)"# + hints + #"(?:\n"# + hints + ")*")
  }()
  private static let fileLine = try! NSRegularExpression(
    pattern: #"^- (.+) \(([^,]+),\s*[\d.]+ KB\), file_id=([0-9a-f-]+)$"#)
  private static let folderLine = try! NSRegularExpression(
    pattern: #"^- (.+) \(folder, (\d+) file\(s\) directly inside\), folder_id=([0-9a-f-]+)$"#)

  static func split(_ content: String) -> (text: String, attachments: [MessageAttachment]) {
    let range = NSRange(content.startIndex..., in: content)
    guard let match = block.firstMatch(in: content, range: range),
      let listRange = Range(match.range(at: 1), in: content),
      let wholeRange = Range(match.range, in: content)
    else { return (content, []) }
    var attachments: [MessageAttachment] = []
    for line in String(content[listRange]).trimmed.split(separator: "\n") {
      let text = String(line).trimmed
      if let groups = captures(folderLine, text) {
        attachments.append(
          .init(kind: .folder, id: groups[2], name: groups[0], detail: "\(groups[1]) files"))
      } else if let groups = captures(fileLine, text) {
        attachments.append(.init(kind: .file, id: groups[2], name: groups[0], detail: groups[1]))
      }
    }
    var text = content
    text.removeSubrange(wholeRange)
    return (text.trimmed, attachments)
  }

  private static func captures(_ regex: NSRegularExpression, _ text: String) -> [String]? {
    guard let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
    else {
      return nil
    }
    return (1..<match.numberOfRanges).map { index in
      Range(match.range(at: index), in: text).map { String(text[$0]) } ?? ""
    }
  }
}
