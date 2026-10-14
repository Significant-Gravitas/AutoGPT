#if UI_PROTOCOL_FIXTURES
  import Foundation

  /// A local stand-in for the chat API that speaks the same UI message stream
  /// protocol. A turn runs on its own clock, independent of any connection,
  /// so leaving, resuming, refreshing and stopping behave as they do against
  /// a server. History is derived from the streamed entries with the same row
  /// rules the app uses, so a reload shows what streamed.
  struct FixtureConversationService: ConversationService {
    enum Scenario: Sendable { case longStream, question }

    static let sessionID = "fixture-session"

    private let backend: FixtureBackend

    init(scenario: Scenario) {
      backend = FixtureBackend(scenario: scenario)
    }

    func detail(sessionID: String, limit: Int, beforeSequence: Int?) async throws -> SessionDetail {
      try await Task.sleep(for: .milliseconds(250))
      return await backend.detail(limit: limit, beforeSequence: beforeSequence)
    }

    func create(expertID: String?) async throws -> CreatedSession {
      CreatedSession(id: Self.sessionID, expertID: nil)
    }

    func openTurn(sessionID: String, message: String, messageID: String) async throws -> TurnOpen {
      try await Task.sleep(for: .milliseconds(300))
      guard let turnID = await backend.startTurn(message: message) else {
        return .refused(
          HTTPError(
            status: 409, body: .string("A reply is already running."), retryAfterSeconds: nil))
      }
      return .accepted(Self.frames(backend, turnID: turnID, after: "0-0"))
    }

    func resumeTurn(sessionID: String, turnID: String, after: String) async throws -> TurnOpen {
      guard await backend.isRunning(turnID) else {
        return .refused(
          HTTPError(
            status: 410, body: .object(["reason": .string("expired")]), retryAfterSeconds: nil))
      }
      return .accepted(Self.frames(backend, turnID: turnID, after: after))
    }

    func cancelTurn(sessionID: String) async throws -> Bool {
      await backend.cancel()
    }

    private static func frames(_ backend: FixtureBackend, turnID: String, after: String)
      -> AsyncThrowingStream<SSEFrame, any Error>
    {
      AsyncThrowingStream { continuation in
        let producer = Task {
          var index = await backend.firstIndex(turnID: turnID, after: after)
          while !Task.isCancelled {
            guard let planned = await backend.planned(turnID: turnID, index: index) else { break }
            let wait = planned.release.timeIntervalSinceNow
            if wait > 0 { try? await Task.sleep(for: .milliseconds(Int(wait * 1000))) }
            if Task.isCancelled { break }
            // A stop can rewrite the rest of the plan while this waits.
            guard let current = await backend.planned(turnID: turnID, index: index),
              current.entry.entryID == planned.entry.entryID
            else { continue }
            continuation.yield(
              .entry(turn: turnID, entryID: current.entry.entryID, chunk: current.entry.chunk))
            index += 1
          }
          continuation.finish()
        }
        continuation.onTermination = { _ in producer.cancel() }
      }
    }
  }

  struct PlannedEntry: Sendable {
    let entry: StreamEntry
    let release: Date
  }

  actor FixtureBackend {
    private var rows: [JSONValue] = []
    private var nextSequence = 1
    private var turnCount = 0
    private var turn: (id: String, startedAt: Date, plan: [PlannedEntry])?
    private let scenario: FixtureConversationService.Scenario

    init(scenario: FixtureConversationService.Scenario) {
      self.scenario = scenario
      let start = Date().addingTimeInterval(-3 * 3600)
      let history = FixtureScripts.history(scenario).enumerated().map { offset, row in
        Self.message(row, sequence: offset + 1, at: start.addingTimeInterval(Double(offset) * 45))
      }
      rows = history
      nextSequence = history.count + 1
    }

    func detail(limit: Int, beforeSequence: Int?) -> SessionDetail {
      settleIfFinished()
      let eligible = rows.filter { row in
        beforeSequence.map { (row["sequence"]?.int ?? 0) < $0 } ?? true
      }
      let window = Array(eligible.suffix(limit))
      let active = turn.map {
        ActiveStream(turnID: $0.id, startedAt: FixtureScripts.iso($0.startedAt), checkpoint: nil)
      }
      return SessionDetail(
        id: FixtureConversationService.sessionID, chatStatus: turn == nil ? "idle" : "running",
        messages: window, activeStream: active, hasMore: eligible.count > window.count,
        oldestSequence: window.first?["sequence"]?.int, expertID: nil)
    }

    func startTurn(message: String) -> String? {
      settleIfFinished()
      guard turn == nil else { return nil }
      let startedAt = Date()
      persist(Row(key: "", role: .user, content: message), at: startedAt.addingTimeInterval(-0.01))
      turnCount += 1
      let id = "fixture-turn-\(turnCount)"
      let plan = FixtureScripts.reply(scenario, to: message, startingAt: startedAt).map {
        PlannedEntry(
          entry: StreamEntry(turn: id, entryID: $0.entry.entryID, chunk: $0.entry.chunk),
          release: $0.release)
      }
      turn = (id, startedAt, plan)
      return id
    }

    func isRunning(_ turnID: String) -> Bool {
      settleIfFinished()
      return turn?.id == turnID
    }

    func firstIndex(turnID: String, after cursor: String) -> Int {
      guard let turn, turn.id == turnID else { return 0 }
      return turn.plan.firstIndex { EntryID.compare($0.entry.entryID, cursor) > 0 }
        ?? turn.plan.count
    }

    func planned(turnID: String, index: Int) -> PlannedEntry? {
      guard let turn, turn.id == turnID, turn.plan.indices.contains(index) else { return nil }
      return turn.plan[index]
    }

    /// Ends the running reply where it is, as the server's cancel does.
    func cancel() -> Bool {
      guard var current = turn else { return false }
      let now = Date()
      var kept = current.plan.filter { $0.release <= now }
      let stamp = UInt64(now.timeIntervalSince1970 * 1000)
      for (offset, type) in ["finish-step", "finish"].enumerated() {
        kept.append(
          PlannedEntry(
            entry: StreamEntry(
              turn: current.id, entryID: "\(stamp)-\(900 + offset)",
              chunk: WireChunk(type: type, fields: ["type": .string(type)])),
            release: now))
      }
      current.plan = kept
      turn = current
      return true
    }

    private func settleIfFinished() {
      guard let current = turn, let last = current.plan.last, last.release <= Date() else { return }
      var log = TurnLog(turnID: current.id, status: .running)
      for planned in current.plan { log.apply(planned.entry) }
      for row in log.rows { persist(row, at: last.release) }
      turn = nil
    }

    private func persist(_ row: Row, at date: Date) {
      rows.append(Self.message(row, sequence: nextSequence, at: date))
      nextSequence += 1
    }

    /// A row as `GET /sessions/{id}` returns it.
    private static func message(_ row: Row, sequence: Int, at date: Date) -> JSONValue {
      var message: [String: JSONValue] = [
        "role": .string(row.role.rawValue), "content": .string(row.content),
        "sequence": .number(Double(sequence)), "created_at": .string(FixtureScripts.iso(date)),
      ]
      if !row.toolCalls.isEmpty {
        message["tool_calls"] = .array(
          row.toolCalls.map { call in
            .object([
              "id": .string(call.id), "type": .string("function"),
              "function": .object([
                "name": .string(call.name), "arguments": .string(call.input.encodedString()),
              ]),
            ])
          })
      }
      if let id = row.toolCallID { message["tool_call_id"] = .string(id) }
      return .object(message)
    }
  }

  enum FixtureScripts {
    static func iso(_ date: Date) -> String {
      let formatter = ISO8601DateFormatter()
      formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
      return formatter.string(from: date)
    }

    static func history(_ scenario: FixtureConversationService.Scenario) -> [Row] {
      switch scenario {
      case .longStream: longHistory()
      case .question: questionHistory()
      }
    }

    private static func longHistory() -> [Row] {
      var rows: [Row] = []
      let topics = [
        "the onboarding emails", "last quarter's churn", "the pricing page test",
        "hiring a support expert",
        "the partner webinar", "our weekly metrics", "the HubSpot cleanup", "a launch checklist",
      ]
      for (index, topic) in (topics + topics + topics).enumerated() {
        rows.append(
          Row(
            key: "", role: .user,
            content: "Can you summarize where we are with \(topic)? (\(index + 1))"))
        var reply = Row(
          key: "", role: .assistant,
          content: """
            Here's where **\(topic)** stands.

            - The draft is ready for review, with two open comments.
            - Results so far are *slightly above* the target.
            - Next step: schedule a check-in with the owning expert.

            Ask me to change anything, or select part of this reply to copy it.
            """)
        if index % 6 == 2 {
          let call = "fixture-search-\(index)"
          reply.toolCalls = [
            ToolCall(
              id: call, name: "web_search", input: .object(["query": .string(topic)]),
              displayName: "Searching the web")
          ]
          rows.append(reply)
          var result = Row(
            key: "", role: .tool,
            content: JSONValue.object([
              "type": .string("web_search"), "message": .string("Found 3 results about \(topic)."),
            ]).encodedString())
          result.toolCallID = call
          rows.append(result)
        } else if index % 6 == 4 {
          let call = "fixture-unknown-\(index)"
          reply.toolCalls = [
            ToolCall(id: call, name: "future_tool", input: .object([:]), displayName: nil)
          ]
          rows.append(reply)
          var result = Row(
            key: "", role: .tool,
            content: JSONValue.object([
              "type": .string("a_card_this_build_does_not_know"),
              "summary": .string("A result from a newer server."),
            ]).encodedString())
          result.toolCallID = call
          rows.append(result)
        } else {
          rows.append(reply)
        }
      }
      return rows
    }

    private static func questionHistory() -> [Row] {
      var ask = Row(
        key: "", role: .assistant,
        content: "Before I set up the HubSpot sync, I need a few details.")
      ask.toolCalls = [
        ToolCall(
          id: "fixture-question", name: "ask_question", input: .object([:]),
          displayName: "Asking a question")
      ]
      var result = Row(key: "", role: .tool, content: questionOutput.encodedString())
      result.toolCallID = "fixture-question"
      return [
        Row(key: "", role: .user, content: "Keep our CRM in sync with the weekly sales summary."),
        ask, result,
      ]
    }

    static let questionOutput = JSONValue.object([
      "type": .string("agent_builder_clarification_needed"),
      "message": .string("Answer these so I can set up the HubSpot sync."),
      "questions": .array([
        .object([
          "question": .string("Which HubSpot records should sync?"), "keyword": .string("records"),
          "options": .array(["Contacts", "Companies", "Deals", "Tickets"].map { .string($0) }),
          "allow_multiple": .bool(true),
        ]),
        .object([
          "question": .string("How often should it run?"), "keyword": .string("frequency"),
          "options": .array(["Every hour", "Every day", "Every week"].map { .string($0) }),
        ]),
        .object([
          "question": .string("Where should the summary go?"), "keyword": .string("destination"),
          "example": .string("For example, #sales in Slack"),
        ]),
      ]),
    ])

    /// The reply as stored stream entries released over time.
    static func reply(
      _ scenario: FixtureConversationService.Scenario, to message: String, startingAt start: Date
    )
      -> [PlannedEntry]
    {
      var entries: [PlannedEntry] = []
      var clock: TimeInterval = 0.2
      let base = UInt64(start.timeIntervalSince1970 * 1000)
      func add(
        _ type: String, _ fields: [String: JSONValue] = [:], after delay: TimeInterval = 0.05
      ) {
        clock += delay
        var all = fields
        all["type"] = .string(type)
        let entryID = "\(base + UInt64(clock * 1000))-\(entries.count)"
        entries.append(
          PlannedEntry(
            entry: StreamEntry(
              turn: "", entryID: entryID, chunk: WireChunk(type: type, fields: all)),
            release: start.addingTimeInterval(clock)))
      }
      func stream(_ text: String, block: String) {
        add("text-start", ["id": .string(block)])
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        var index = 0
        while index < words.count {
          let piece = words[index..<min(index + 3, words.count)].joined(separator: " ")
          index += 3
          add(
            "text-delta",
            ["id": .string(block), "delta": .string(piece + (index < words.count ? " " : ""))],
            after: 0.06)
        }
        add("text-end", ["id": .string(block)])
      }
      add("start", ["messageId": .string(UUID().uuidString)])
      add("start-step")
      switch scenario {
      case .longStream:
        add("reasoning-start", ["id": .string("thinking")])
        for thought in [
          "Reviewing the request. ", "Checking the plan against the budget. ",
          "Drafting the reply.",
        ] {
          add(
            "reasoning-delta", ["id": .string("thinking"), "delta": .string(thought)], after: 0.3)
        }
        add("reasoning-end", ["id": .string("thinking")])
        add("data-status", ["data": .object(["message": .string("Otto is drafting the plan…")])])
        stream(longReply, block: "answer")
        add(
          "tool-input-start",
          ["toolCallId": .string("fixture-live-search"), "toolName": .string("web_search")])
        add(
          "data-tool-display",
          [
            "data": .object([
              "toolCallId": .string("fixture-live-search"),
              "displayName": .string("Searching the web"),
            ])
          ])
        add(
          "tool-input-available",
          [
            "toolCallId": .string("fixture-live-search"), "toolName": .string("web_search"),
            "input": .object(["query": .string("spring campaign benchmarks")]),
          ])
        add(
          "tool-output-available",
          [
            "toolCallId": .string("fixture-live-search"),
            "output": .object([
              "type": .string("web_search"), "message": .string("Found 3 benchmark reports."),
            ]),
          ], after: 1.2)
        stream("Benchmarks agree with the plan above. Tell me what to change.", block: "follow-up")
      case .question:
        stream(
          "Thanks. This fixture received your answers:\n\n\(message)\n\nIn the integrated app, Otto continues the setup from here.",
          block: "answer")
      }
      add("finish-step")
      add("finish")
      return entries
    }

    private static let longReply = """
      ## Launch plan for the spring campaign

      Here's a plan that keeps the team focused and the budget visible. **Each step has an owner**, and anything that needs your approval is called out.

      1. Audit last quarter's results and pull the three best-performing channels.
      2. Draft the messaging brief and share it with the design expert.
      3. Schedule the email sequence for the first Tuesday of the month.
      4. Review spend weekly and pause anything below a 2× return.

      > Tip: select part of any paragraph to copy just that sentence.

      ### Budget split

      | Channel | Share | Owner |
      | --- | --- | --- |
      | Email | 40% | Growth expert |
      | Paid social | 35% | Ads expert |
      | Partnerships | 25% | Otto |

      ### Automation

      The weekly report runs this script:

      ```python
      def weekly_report(rows):
          total = sum(row["spend"] for row in rows)
          return {"total": total, "channels": len(rows)}
      ```

      Week one focuses on the audit, so nothing goes out yet. Week two drafts the brief and the first two emails. Week three is review: every expert comments, and you approve the final copy. Week four launches, and the report starts the following Monday.

      If a channel underperforms for two weeks in a row, I'll propose moving its share to the best channel and ask before changing anything. Read more in the [campaign checklist](https://agpt.co) or ask me to adjust any step. ~~The old plan~~ is retired, and *everything here* can change.
      """
  }

  /// Local connections for the editor fixture. `rejected-key` as a new
  /// secret simulates the provider refusing it.
  struct FixtureConnectionService: ConnectionService {
    private let store = FixtureConnectionStore()

    var canEdit: Bool { true }

    func list() async throws -> [ConnectionSummary] {
      try await Task.sleep(for: .milliseconds(300))
      return await store.connections
    }

    func update(_ id: String, _ change: ConnectionChange) async throws -> ConnectionEditOutcome {
      try await Task.sleep(for: .milliseconds(600))
      if change.replacementSecret?.trimmed == "rejected-key" {
        return .rejected("HubSpot rejected this key. Check it and try again.")
      }
      guard let updated = await store.rename(id, to: change.title) else {
        return .rejected("This connection no longer exists.")
      }
      return .saved(updated)
    }
  }

  actor FixtureConnectionStore {
    private(set) var connections = [
      ConnectionSummary(
        id: "fixture-hubspot", provider: "hubspot", kind: .apiKey, title: "HubSpot (sales team)",
        serviceName: "HubSpot", host: nil, username: nil, scopes: [], isManaged: false),
      ConnectionSummary(
        id: "fixture-github", provider: "github", kind: .oauth2, title: "GitHub",
        serviceName: "GitHub",
        host: nil, username: "octocat", scopes: ["repo", "read:org"], isManaged: false),
      ConnectionSummary(
        id: "fixture-internal-api", provider: "http", kind: .hostScoped, title: "Internal API",
        serviceName: "HTTP request", host: "api.internal.example", username: nil, scopes: [],
        isManaged: false),
    ]

    func rename(_ id: String, to title: String) -> ConnectionSummary? {
      guard let index = connections.firstIndex(where: { $0.id == id }) else { return nil }
      let current = connections[index]
      let renamed = ConnectionSummary(
        id: current.id, provider: current.provider, kind: current.kind, title: title,
        serviceName: current.serviceName, host: current.host, username: current.username,
        scopes: current.scopes, isManaged: current.isManaged)
      connections[index] = renamed
      return renamed
    }
  }
#endif
