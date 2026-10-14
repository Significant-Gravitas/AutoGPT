import Foundation

/// Presentation families from docs/engineering/native-mobile/card-catalog.md.
/// `actionable` marks families whose cards carry user actions on the web.
/// Families without a native card here say plainly that their actions
/// aren't in the app yet; the generic details view is a compatibility
/// fallback, not an implementation of a card.
struct CardFamily: Equatable, Sendable {
  let id: String
  let label: String
  let symbol: String
  let actionable: Bool
}

enum CardCatalog {
  static let status = CardFamily(
    id: "status", label: "Status", symbol: "info.circle", actionable: false)
  static let access = CardFamily(
    id: "access", label: "Sign-in needed", symbol: "lock", actionable: true)
  static let workflow = CardFamily(
    id: "workflow", label: "Workflow", symbol: "bolt", actionable: true)
  static let connection = CardFamily(
    id: "connection", label: "Connection", symbol: "key", actionable: true)
  static let construction = CardFamily(
    id: "construction", label: "Workflow building", symbol: "wrench.and.screwdriver",
    actionable: true)
  static let question = CardFamily(
    id: "question", label: "Question", symbol: "questionmark.circle", actionable: true)
  static let capability = CardFamily(
    id: "capability", label: "Capability", symbol: "puzzlepiece.extension", actionable: true)
  static let approval = CardFamily(
    id: "approval", label: "Approval", symbol: "hand.raised", actionable: true)
  static let schedule = CardFamily(
    id: "schedule", label: "Schedule", symbol: "calendar", actionable: true)
  static let expertResource = CardFamily(
    id: "expertResource", label: "Expert resources", symbol: "person.2", actionable: true)
  static let delegation = CardFamily(
    id: "delegation", label: "Delegated work", symbol: "arrow.triangle.2.circlepath",
    actionable: true)
  static let docs = CardFamily(
    id: "docs", label: "Web and docs", symbol: "globe", actionable: false)
  static let files = CardFamily(id: "files", label: "Files", symbol: "folder", actionable: true)
  static let computer = CardFamily(
    id: "computer", label: "Computer", symbol: "laptopcomputer", actionable: true)
  static let memory = CardFamily(
    id: "memory", label: "Memory and plans", symbol: "brain", actionable: false)
  static let memoryForget = CardFamily(
    id: "memoryForget", label: "Memory", symbol: "brain", actionable: true)
  static let channels = CardFamily(
    id: "channels", label: "Channels and skills", symbol: "text.bubble", actionable: false)
  static let company = CardFamily(
    id: "company", label: "AI company", symbol: "person.3", actionable: true)
  static let historical = CardFamily(
    id: "historical", label: "Earlier result", symbol: "clock", actionable: false)

  /// Every `type` the backend's tool response models declare, by family.
  static let responseTypes: [String: CardFamily] = [
    "error": status, "no_results": status, "need_login": access,
    "input_validation_error": status, "understanding_updated": status, "platform_info": status,
    "agents_found": workflow, "agent_details": workflow, "execution_started": workflow,
    "agent_output": workflow, "setup_requirements": connection,
    "credential_grant_requested": connection, "trigger_config_required": connection,
    "suggested_goal": construction, "agent_builder_guide": construction,
    "agent_builder_preview": construction, "agent_builder_saved": construction,
    "agent_builder_validation_result": construction, "agent_builder_fix_result": construction,
    "task_decomposition": construction, "agent_builder_clarification_needed": question,
    "expert_onboarding": question, "block_list": capability, "block_details": capability,
    "block_output": capability, "capability_list": capability, "capability_details": capability,
    "review_required": approval, "approval_required": approval, "schedule_list": schedule,
    "schedule_deleted": schedule, "schedule_toggled": schedule, "schedule_created": schedule,
    "routines": schedule, "routine": schedule, "agent_trigger_list": schedule,
    "trigger_setup": schedule, "preset_list": schedule, "preset_updated": schedule,
    "preset_deleted": schedule, "expert_workflow": expertResource,
    "expert_workflows": expertResource, "expert_credentials": expertResource,
    "mcp_tools_discovered": delegation, "mcp_tool_output": delegation,
    "doc_search_results": docs, "doc_page": docs, "web_fetch": docs, "web_search": docs,
    "feature_request_search": docs, "feature_request_created": docs,
    "workspace_file_list": files, "workspace_file_content": files,
    "workspace_file_metadata": files, "workspace_file_written": files,
    "workspace_file_deleted": files, "folder_created": files, "folder_list": files,
    "folder_updated": files, "folder_moved": files, "folder_deleted": files,
    "agents_moved_to_folder": files, "browser_navigate": computer, "browser_act": computer,
    "browser_screenshot": computer, "bash_exec": computer, "desktop_stream": computer,
    "memory_store": memory, "memory_search": memory, "memory_forget_candidates": memoryForget,
    "memory_forget_confirm": memoryForget, "todo_write": memory,
    "chat_platform_channel_list": channels, "chat_platform_posted": channels,
    "chat_platform_edited": channels, "skill_stored": channels, "skill_loaded": channels,
    "skill_deleted": channels, "skill_list": channels, "expert_soul_updated": company,
    "expert_change_proposed": company, "expert_change_applied": company, "team_roster": company,
    "expert_chat_list": company, "expert_chat_transcript": company, "team_consult": company,
    "session_list": company, "session_message": company, "mcp_guide": historical,
  ]

  /// Families with a native card in this build. Everything else uses the
  /// read-only details view and says its actions aren't available.
  static let nativeCards: Set<String> = [question.id]
}

struct ToolPresentation {
  let title: String
  let statusLabel: String
  let symbol: String
  /// The output's `type` discriminator, when it has one.
  let responseType: String?
  let family: CardFamily?
  let failed: Bool
  let summary: String?
  let coverageNote: String

  private static let summaryFields = [
    "message", "error", "summary", "text", "result", "content", "title",
  ]

  init(_ part: ToolPart) {
    let output = part.output?.object
    let responseType = part.output?["type"]?.nonEmptyString
    let family = responseType.flatMap { CardCatalog.responseTypes[$0] }
    let failed =
      part.state == .failed || responseType == "error" || responseType == "input_validation_error"
      || output?["success"]?.bool == false
    let state: ToolPart.State = failed && part.state == .done ? .failed : part.state
    self.title = part.displayName ?? Self.humanize(part.toolName)
    self.statusLabel = Self.label(state)
    self.symbol =
      state == .failed ? "exclamationmark.triangle" : family?.symbol ?? "wrench.and.screwdriver"
    self.responseType = responseType
    self.family = family
    self.failed = state == .failed
    self.summary = Self.summarize(part.output)
    self.coverageNote = Self.coverageNote(part, responseType, family)
  }

  private static func label(_ state: ToolPart.State) -> String {
    switch state {
    case .running: "Working…"
    case .done: "Done"
    case .failed: "Failed"
    case .interrupted: "Interrupted"
    case .unrecorded: "No result recorded"
    }
  }

  private static func coverageNote(_ part: ToolPart, _ responseType: String?, _ family: CardFamily?)
    -> String
  {
    if part.state == .running { return "Still running. The result appears here when it finishes." }
    guard part.output != nil else { return "No result was saved for this step." }
    guard responseType != nil else {
      return "This step returned plain output. It's shown below as text."
    }
    guard let family else {
      return
        "This result type isn't recognized by this version of the app, so only its raw details are shown and none of its actions are offered."
    }
    if CardCatalog.nativeCards.contains(family.id) {
      return "This step has a native card in the conversation. Its raw details are shown here."
    }
    return family.actionable
      ? "The native \(family.label.lowercased()) card isn't built yet, so its actions aren't available in the app. Its details are shown read-only."
      : "The native \(family.label.lowercased()) card isn't built yet. Its details are shown read-only."
  }

  /// `run_agent` → "Run agent"; `mcp__server__tool` → "Tool (server)".
  static func humanize(_ name: String) -> String {
    guard !name.isEmpty else { return "Tool" }
    if name.hasPrefix("mcp__") {
      let pieces = name.components(separatedBy: "__")
      let server = pieces.count > 1 ? pieces[1] : ""
      let rest = pieces.count > 2 ? Array(pieces[2...]) : []
      let tool = rest.isEmpty ? server : rest.joined(separator: " ")
      return sentenceCase(tool) + (!rest.isEmpty && !server.isEmpty ? " (\(server))" : "")
    }
    return sentenceCase(name)
  }

  private static func sentenceCase(_ raw: String) -> String {
    let spaced =
      raw
      .replacingOccurrences(of: #"([a-z0-9])([A-Z])"#, with: "$1 $2", options: .regularExpression)
      .replacingOccurrences(of: #"[_-]+"#, with: " ", options: .regularExpression)
      .trimmed.lowercased()
    guard let first = spaced.first else { return "Tool" }
    return first.uppercased() + spaced.dropFirst()
  }

  private static func summarize(_ output: JSONValue?) -> String? {
    if let text = output?.string {
      let trimmed = text.trimmed
      if trimmed.isEmpty { return nil }
      if trimmed.hasPrefix("{") || trimmed.hasPrefix("[") { return "The result couldn't be read." }
      return collapse(trimmed)
    }
    guard let record = output?.object else { return nil }
    for field in summaryFields {
      if let value = record[field]?.nonEmptyString { return collapse(value) }
    }
    return nil
  }

  private static func collapse(_ text: String) -> String {
    text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression).truncated(
      to: 240)
  }
}

enum BoundedJSON {
  private static let secretKey = try! NSRegularExpression(
    pattern:
      #"(pass(word)?|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key|session[_-]?id)"#,
    options: .caseInsensitive)

  /// Pretty-prints untrusted tool data for a details view. Secret-looking keys
  /// are masked and every dimension is capped, so a huge or hostile payload
  /// stays readable and never leaks a credential into a screenshot.
  static func prettyPrint(
    _ value: JSONValue, maxChars: Int = 20_000, maxDepth: Int = 6, maxItems: Int = 60,
    maxString: Int = 2_000
  ) -> String {
    func walk(_ node: JSONValue, _ depth: Int) -> JSONValue {
      switch node {
      case .string(let text):
        return .string(text.truncated(to: maxString))
      case .array(let items):
        if depth >= maxDepth { return .string("[…]") }
        var shaped = items.prefix(maxItems).map { walk($0, depth + 1) }
        if items.count > maxItems { shaped.append(.string("… \(items.count - maxItems) more")) }
        return .array(shaped)
      case .object(let fields):
        if depth >= maxDepth { return .string("{…}") }
        var shaped: [String: JSONValue] = [:]
        for key in fields.keys.sorted().prefix(maxItems) {
          let masked =
            secretKey.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)) != nil
          shaped[key] = masked ? .string("••••••") : walk(fields[key] ?? .null, depth + 1)
        }
        if fields.count > maxItems {
          shaped["…"] = .string("\(fields.count - maxItems) more fields")
        }
        return .object(shaped)
      default:
        return node
      }
    }
    let shaped = walk(value, 0)
    let text = shaped.string ?? shaped.encodedString(pretty: true)
    return text.truncated(to: maxChars)
  }
}
