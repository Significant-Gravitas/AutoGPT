import SwiftUI

/// Callbacks shared by every row of a conversation. Rows compare equal by
/// their content only; these always route to the same controller.
struct ConversationActions {
  let retrySend: (String) -> Void
  let discard: (String) -> Void
  let openTool: (ToolPart) -> Void
  let answer: (String) -> Void
  let loadOlder: () -> Void
  let refresh: () -> Void
}

struct ConversationView: View {
  let controller: ConversationController
  let participant: (String?) -> Participant
  var onAppear: () -> Void = {}
  @State private var toolDetails: ToolPart?
  @State private var sends = 0

  var body: some View {
    let snapshot = controller.snapshot
    let person = participant(snapshot.expertID)
    content(snapshot, person)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .background(Color(uiColor: .systemBackground))
      .safeAreaInset(edge: .bottom, spacing: 0) {
        ComposerView(
          placeholder: "Message \(person.name)",
          readOnlyReason: person.readOnlyReason,
          busy: snapshot.turn != .idle || snapshot.loadState != .ready,
          canStop: snapshot.canStop,
          onSend: { text in
            sends += 1
            Task { await controller.send(text) }
          },
          onStop: { Task { await controller.stop() } })
      }
      .navigationTitle(person.name)
      .navigationBarTitleDisplayMode(.inline)
      .toolbar(.hidden, for: .tabBar)
      .toolbar {
        ToolbarItem(placement: .principal) {
          HStack(spacing: 8) {
            AvatarView(
              name: person.name, url: person.avatarURL, size: 32, asset: person.avatarAsset)
            VStack(alignment: .leading, spacing: 1) {
              Text(person.name).font(.headline)
              if let role = person.role {
                Text(role).font(.caption).foregroundStyle(.secondary).lineLimit(1)
              }
            }
          }
          .accessibilityElement(children: .combine)
          .accessibilityAddTraits(.isHeader)
        }
      }
      .onAppear {
        controller.start()
        onAppear()
      }
      .sheet(item: $toolDetails) { part in
        ToolDetailsView(part: part)
      }
  }

  @ViewBuilder
  private func content(_ snapshot: ConversationSnapshot, _ person: Participant) -> some View {
    switch snapshot.loadState {
    case .loading:
      ProgressView("Loading conversation")
    case .error:
      ContentUnavailableView {
        Label("Couldn't open this conversation", systemImage: "exclamationmark.bubble")
      } description: {
        Text(snapshot.loadError ?? "Couldn't open this conversation.")
      } actions: {
        Button("Try Again") { controller.reload() }
      }
    case .ready:
      if snapshot.items.isEmpty && snapshot.turn == .idle && snapshot.notice == nil {
        ConversationIntroView(participant: person)
      } else {
        MessageListView(
          snapshot: snapshot, participant: person, sends: sends,
          actions: ConversationActions(
            retrySend: { id in Task { await controller.retry(id) } },
            discard: { controller.discard($0) },
            openTool: { toolDetails = $0 },
            answer: { text in
              sends += 1
              Task { await controller.send(text) }
            },
            loadOlder: { Task { await controller.loadOlder() } },
            refresh: { Task { await controller.refresh() } }))
      }
    }
  }
}

/// Newest message at the bottom. The list follows new content while the
/// reader is at the end, and leaves older reading undisturbed otherwise.
struct MessageListView: View {
  let snapshot: ConversationSnapshot
  let participant: Participant
  let sends: Int
  let actions: ConversationActions
  @State private var atEnd = true
  @State private var followingReply = true
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  private static let endID = "conversation-end"

  var body: some View {
    let canRetry = snapshot.turn == .idle && snapshot.loadState == .ready
    let questionsPendingFor = pendingQuestionItem
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 0) {
          if snapshot.loadingOlder {
            ProgressView().frame(maxWidth: .infinity).padding()
          } else if snapshot.hasOlder {
            Color.clear.frame(height: 1).onAppear(perform: actions.loadOlder)
          }
          ForEach(snapshot.items) { item in
            row(item, canRetry: canRetry, questionsPending: item.id == questionsPendingFor)
              .id(item.id)
          }
          TurnStatusView(
            turn: snapshot.turn, statusText: snapshot.statusText, notice: snapshot.notice,
            participantName: participant.name, onRetry: actions.refresh)
          Color.clear
            .frame(height: 1)
            .id(Self.endID)
            .onAppear {
              if #unavailable(iOS 18.0) {
                atEnd = true
                followingReply = true
              }
            }
            .onDisappear {
              if #unavailable(iOS 18.0) { atEnd = false }
            }
        }
        .padding(.vertical, 8)
      }
      .defaultScrollAnchor(.bottom)
      .alignShortConversationAtTop()
      .scrollDismissesKeyboard(.interactively)
      .refreshable { actions.refresh() }
      .accessibilityIdentifier("message-list")
      .trackConversationScroll(
        atEndChanged: { visible in
          atEnd = visible
          if visible { followingReply = true }
        },
        interactionChanged: { interacting in followingReply = interacting ? false : atEnd }
      )
      .onChange(of: snapshot.revision) {
        if followingReply { scrollToEnd(proxy, animated: false) }
      }
      .onChange(of: sends) {
        followingReply = true
        scrollToEnd(proxy, animated: true)
      }
      .onChange(of: snapshot.items.first?.id) { previous, _ in
        // Older messages arrived above: keep the reader where they were.
        if let previous, !atEnd { proxy.scrollTo(previous, anchor: .top) }
      }
      .overlay(alignment: .bottomTrailing) {
        if !atEnd {
          Button {
            followingReply = true
            scrollToEnd(proxy, animated: true)
          } label: {
            Image(systemName: "arrow.down")
              .font(.body.weight(.semibold))
              .frame(width: 44, height: 44)
              .background(.regularMaterial, in: Circle())
              .overlay(Circle().strokeBorder(.quaternary, lineWidth: 0.5))
          }
          .buttonStyle(.plain)
          .accessibilityLabel("Latest messages")
          .accessibilityIdentifier("latest-messages")
          .padding(12)
        }
      }
    }
  }

  /// Questions wait for an answer only while the asking reply is the latest
  /// message and nothing is running.
  private var pendingQuestionItem: String? {
    guard snapshot.turn == .idle, snapshot.loadState == .ready, let last = snapshot.items.last,
      case .assistant(let item) = last
    else { return nil }
    return item.key
  }

  private func scrollToEnd(_ proxy: ScrollViewProxy, animated: Bool) {
    if animated && !reduceMotion {
      withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(Self.endID, anchor: .bottom) }
    } else {
      proxy.scrollTo(Self.endID, anchor: .bottom)
    }
  }

  @ViewBuilder
  private func row(_ item: TimelineItem, canRetry: Bool, questionsPending: Bool) -> some View {
    switch item {
    case .user(let user):
      UserMessageView(item: user, canRetry: canRetry, actions: actions).equatable()
    case .assistant(let assistant):
      AssistantMessageView(
        item: assistant, participant: participant, questionsPending: questionsPending,
        actions: actions
      )
      .equatable()
    case .work(_, let text):
      WorkUpdateView(text: text)
    }
  }
}

extension View {
  fileprivate func alignShortConversationAtTop() -> some View {
    Group {
      if #available(iOS 18.0, *) {
        defaultScrollAnchor(.top, for: .alignment)
      } else {
        self
      }
    }
  }

  @ViewBuilder
  fileprivate func trackConversationScroll(
    atEndChanged: @escaping (Bool) -> Void,
    interactionChanged: @escaping (Bool) -> Void
  ) -> some View {
    if #available(iOS 18.0, *) {
      onScrollGeometryChange(for: Bool.self) { geometry in
        geometry.visibleRect.maxY >= geometry.contentSize.height - 24
      } action: { _, visible in
        atEndChanged(visible)
      }
      .onScrollPhaseChange { _, phase in
        if phase == .interacting { interactionChanged(true) }
        if phase == .idle { interactionChanged(false) }
      }
    } else {
      simultaneousGesture(
        DragGesture(minimumDistance: 10)
          .onChanged { _ in interactionChanged(true) }
          .onEnded { _ in interactionChanged(false) })
    }
  }
}

struct UserMessageView: View, Equatable {
  let item: UserItem
  let canRetry: Bool
  let actions: ConversationActions
  @Environment(\.dynamicTypeSize) private var typeSize

  nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.item == rhs.item && lhs.canRetry == rhs.canRetry
  }

  var body: some View {
    let local = item.local
    VStack(alignment: .trailing, spacing: 4) {
      VStack(alignment: .leading, spacing: 4) {
        if !item.text.isEmpty {
          SelectableText(
            text: NSAttributedString(
              string: item.text,
              attributes: [
                .font: TextStyleFont.font(.body, typeSize), .foregroundColor: UIColor.label,
              ]),
            hugsContent: true)
        }
        ForEach(item.attachments) { attachment in
          Label(attachment.name, systemImage: attachment.kind == .folder ? "folder" : "doc")
            .font(.footnote)
            .foregroundStyle(.primary)
            .lineLimit(1)
            .accessibilityLabel("Attachment \(attachment.name)")
        }
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 10)
      .background(Brand.surface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
      .opacity(local?.state == .sending ? 0.6 : 1)
      if local?.state == .sending {
        Text("Sending…").font(.caption).foregroundStyle(.secondary)
      }
      if let local, local.state == .failed {
        VStack(alignment: .trailing, spacing: 2) {
          Text("Not sent. \(local.error ?? "")")
            .font(.footnote)
            .foregroundStyle(.red)
            .multilineTextAlignment(.trailing)
          HStack {
            Button("Delete") { actions.discard(local.messageID) }
            Button("Retry") { actions.retrySend(local.messageID) }
              .disabled(!canRetry)
          }
          .buttonStyle(.borderless)
        }
        .accessibilityElement(children: .contain)
      }
    }
    .padding(.leading, 48)
    .frame(maxWidth: .infinity, alignment: .trailing)
    .padding(.horizontal, 16)
    .padding(.vertical, 4)
  }
}

struct AssistantMessageView: View, Equatable {
  let item: AssistantItem
  let participant: Participant
  let questionsPending: Bool
  let actions: ConversationActions

  nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.item == rhs.item && lhs.participant == rhs.participant
      && lhs.questionsPending == rhs.questionsPending
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 8) {
        AvatarView(
          name: participant.name, url: participant.avatarURL, size: 28,
          asset: participant.avatarAsset)
        Text(participant.name).font(.subheadline.weight(.semibold))
      }
      .accessibilityElement(children: .combine)
      .accessibilityAddTraits(.isHeader)
      .padding(.bottom, 4)
      ForEach(item.parts) { part in
        partView(part)
      }
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 16)
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  @ViewBuilder
  private func partView(_ part: AssistantPart) -> some View {
    switch part {
    case .text(_, let text, _):
      MarkdownView(text: text)
    case .reasoning(_, let text, let streaming):
      ReasoningView(text: text, streaming: streaming)
    case .tool(_, let tool):
      if let questions = QuestionSet(tool) {
        QuestionCardView(
          toolCallID: tool.toolCallID, questions: questions, pending: questionsPending,
          onSubmit: actions.answer, onDetails: { actions.openTool(tool) })
      } else {
        ToolRowView(tool: tool) { actions.openTool(tool) }
      }
    case .error(_, let text, let retryable):
      Label {
        Text(text + (retryable ? " Sending your message again may work." : ""))
          .font(.subheadline)
          .textSelection(.enabled)
      } icon: {
        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
      }
      .padding(12)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Brand.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    case .system(_, let text):
      Text(text)
        .font(.footnote)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
    }
  }
}

private struct ReasoningView: View {
  let text: String
  let streaming: Bool
  @State private var expanded = false

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Button {
        expanded.toggle()
      } label: {
        HStack(spacing: 4) {
          Image(systemName: "brain").font(.footnote)
          Text(streaming ? "Thinking…" : "Thought process").font(.footnote)
          Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption2)
        }
        .foregroundStyle(.secondary)
        .frame(minHeight: 44)
      }
      .buttonStyle(.plain)
      .accessibilityLabel(streaming ? "Thinking" : "Thought process")
      .accessibilityValue(expanded ? "Expanded" : "Collapsed")
      if expanded {
        MarkdownView(text: text, secondary: true)
      }
    }
  }
}

/// One tool step as a compact, tappable row that opens its details. A row
/// offers no card action; families without a native card say so in details.
struct ToolRowView: View {
  let tool: ToolPart
  let onOpen: () -> Void

  var body: some View {
    let presentation = ToolPresentation(tool)
    Button(action: onOpen) {
      HStack(spacing: 8) {
        if tool.state == .running {
          ProgressView()
        } else {
          Image(systemName: presentation.symbol)
            .foregroundStyle(presentation.failed ? Color.red : Color.secondary)
        }
        VStack(alignment: .leading, spacing: 2) {
          Text(presentation.title).font(.subheadline).foregroundStyle(.primary).lineLimit(1)
          if let summary = presentation.summary, tool.state != .running {
            Text(summary).font(.footnote).foregroundStyle(.secondary).lineLimit(2)
          }
        }
        Spacer(minLength: 4)
        Text(presentation.statusLabel)
          .font(.caption)
          .foregroundStyle(
            presentation.failed ? AnyShapeStyle(Color.red) : AnyShapeStyle(.tertiary))
        Image(systemName: "chevron.right")
          .font(.caption.weight(.semibold))
          .foregroundStyle(.tertiary)
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 8)
      .frame(minHeight: 44)
      .background(Brand.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
    .buttonStyle(.plain)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("\(presentation.title), \(presentation.statusLabel)")
    .accessibilityHint("Shows this step's details")
    .accessibilityAddTraits(.isButton)
  }
}

struct WorkUpdateView: View {
  let text: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Label("Work update", systemImage: "checklist")
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.secondary)
      MarkdownView(text: text)
    }
    .padding(12)
    .background(Brand.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    .padding(.horizontal, 16)
    .padding(.vertical, 8)
  }
}

/// Sits under the newest message: what the reply is doing, or what went wrong.
struct TurnStatusView: View {
  let turn: TurnPhase
  let statusText: String?
  let notice: String?
  let participantName: String
  let onRetry: () -> Void

  var body: some View {
    let label = phaseLabel
    if label == nil && notice == nil {
      Color.clear.frame(height: 12)
    } else {
      VStack(spacing: 6) {
        if let label {
          HStack(spacing: 8) {
            if turn != .offline { ProgressView() }
            Text(label)
              .font(.footnote)
              .foregroundStyle(.secondary)
              .accessibilityIdentifier("turn-status")
          }
        }
        if let notice {
          Text(notice).font(.footnote).foregroundStyle(.red).multilineTextAlignment(.center)
        }
        if turn == .offline || (notice != nil && turn == .idle) {
          Button(turn == .offline ? "Reconnect" : "Try Again", action: onRetry)
            .buttonStyle(.borderless)
        }
      }
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 16)
      .padding(.vertical, 12)
      .accessibilityElement(children: .contain)
    }
  }

  private var phaseLabel: String? {
    switch turn {
    case .idle: nil
    case .sending: "Sending…"
    case .streaming: statusText ?? "\(participantName) is working…"
    case .reconnecting: "Reconnecting…"
    case .stopping: "Stopping…"
    case .queued: "Waiting for a free slot…"
    case .offline: "Not connected. The reply keeps going on the server."
    }
  }
}

/// The empty state of a conversation that has no messages yet.
struct ConversationIntroView: View {
  let participant: Participant

  var body: some View {
    GeometryReader { geometry in
      ScrollView {
        VStack(spacing: 12) {
          if participant.expertID == nil {
            OttoArtwork(size: 120)
          } else {
            AvatarView(
              name: participant.name, url: participant.avatarURL, size: 96,
              asset: participant.avatarAsset)
          }
          Text(
            participant.expertID == nil
              ? "What are we working on?" : "Talk with \(participant.name)"
          )
          .font(.title2.bold())
          .padding(.top, 8)
          .accessibilityAddTraits(.isHeader)
          Text(
            participant.readOnlyReason
              ?? (participant.expertID == nil
                ? "A task, an idea, or a busy day.\nLet's make room for what matters."
                : participant.role ?? "Start a conversation with \(participant.name).")
          )
          .foregroundStyle(.secondary)
        }
        .multilineTextAlignment(.center)
        .padding(32)
        .frame(maxWidth: .infinity)
        .frame(minHeight: geometry.size.height)
        .contentShape(Rectangle())
      }
      .scrollDismissesKeyboard(.interactively)
      .onTapGesture {
        UIApplication.shared.sendAction(
          #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
      }
    }
  }
}

struct ComposerView: View {
  let placeholder: String
  /// Why nothing can be sent at all (e.g. a let-go expert). Hides the input.
  let readOnlyReason: String?
  /// A reply is in progress: sending waits for it, and Stop is offered.
  let busy: Bool
  let canStop: Bool
  let onSend: (String) -> Void
  let onStop: () -> Void
  @State private var text = ""
  @State private var sent = 0
  @FocusState private var focused: Bool

  private var canSend: Bool { !busy && !text.trimmed.isEmpty }

  private var inputHint: String {
    busy ? "You can send when the current reply finishes" : ""
  }

  var body: some View {
    VStack(spacing: 0) {
      if let readOnlyReason {
        Text(readOnlyReason)
          .font(.footnote)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
          .frame(maxWidth: .infinity)
          .padding(16)
      } else {
        HStack(alignment: .bottom, spacing: 4) {
          TextField("Message", text: $text, prompt: Text(placeholder), axis: .vertical)
            .lineLimit(1...6)
            .focused($focused)
            .padding(.vertical, 11)
            .padding(.leading, 14)
            .accessibilityLabel("Message")
            .accessibilityHint(inputHint)
            .accessibilityIdentifier("composer-input")
          if busy && canStop {
            Button(action: onStop) {
              Image(systemName: "stop.circle.fill")
                .font(.system(size: 30))
                .foregroundStyle(.primary)
                .frame(width: 44, height: 44)
            }
            .accessibilityLabel("Stop reply")
            .accessibilityIdentifier("composer-stop")
          } else {
            Button(action: send) {
              Image(systemName: "arrow.up.circle.fill")
                .font(.system(size: 30))
                .foregroundStyle(canSend ? AnyShapeStyle(.tint) : AnyShapeStyle(.tertiary))
                .frame(width: 44, height: 44)
            }
            .disabled(!canSend)
            .accessibilityLabel("Send")
            .accessibilityIdentifier("composer-send")
          }
        }
        .background(
          Color(uiColor: .systemBackground),
          in: RoundedRectangle(cornerRadius: 24, style: .continuous)
        )
        .overlay(
          RoundedRectangle(cornerRadius: 24, style: .continuous)
            .strokeBorder(Color(uiColor: .separator), lineWidth: 0.5)
        )
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
      }
    }
    .background(.bar)
    .sensoryFeedback(.impact(weight: .light), trigger: sent)
  }

  private func send() {
    guard canSend else { return }
    onSend(text)
    text = ""
    sent += 1
  }
}
