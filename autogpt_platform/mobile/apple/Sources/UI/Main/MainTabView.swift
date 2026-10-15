import SwiftUI

/// The two primary destinations: conversations, and the AI company you talk to.
struct MainTabView: View {
  let context: SignedInContext
  let session: SessionStore
  @State private var chats: ChatListStore
  @State private var experts: ExpertDirectory
  @State private var chatsNavigator = ConversationNavigator()
  @State private var companyNavigator = ConversationNavigator()
  @State private var tab = Tab.chats
  @State private var showingAccount = false
  @Environment(\.scenePhase) private var scenePhase

  private enum Tab: Hashable { case chats, company }

  init(context: SignedInContext, session: SessionStore) {
    self.context = context
    self.session = session
    _chats = State(initialValue: ChatListStore(api: context.api))
    _experts = State(initialValue: ExpertDirectory(api: context.api))
  }

  var body: some View {
    TabView(selection: $tab) {
      stack(chatsNavigator) {
        ChatsView(
          context: context, store: chats, experts: experts, open: chatsNavigator.open,
          showAccount: { showingAccount = true })
      }
      .tabItem { Label("Chats", systemImage: "bubble.left.and.bubble.right") }
      .tag(Tab.chats)
      stack(companyNavigator) {
        CompanyView(
          context: context, experts: experts, open: companyNavigator.open,
          showAccount: { showingAccount = true })
      }
      .tabItem { Label("Company", systemImage: "person.3") }
      .tag(Tab.company)
    }
    .toolbar(
      chatsNavigator.path.isEmpty && companyNavigator.path.isEmpty ? .visible : .hidden,
      for: .tabBar
    )
    .sheet(isPresented: $showingAccount) {
      AccountView(context: context, session: session)
    }
    .task { experts.ensureLoaded() }
    .onChange(of: scenePhase) { _, phase in
      for controller in chatsNavigator.openControllers + companyNavigator.openControllers {
        controller.scenePhaseChanged(phase)
      }
    }
    .onDisappear {
      chatsNavigator.disposeAll()
      companyNavigator.disposeAll()
    }
  }

  private func stack<Root: View>(_ navigator: ConversationNavigator, @ViewBuilder root: () -> Root)
    -> some View
  {
    NavigationStack(path: Binding(get: { navigator.path }, set: { navigator.setPath($0) })) {
      root()
        .navigationDestination(for: ConversationRoute.self) { route in
          let controller = navigator.controller(
            for: route, service: LiveConversationService(api: context.api))
          ConversationView(
            controller: controller,
            participant: {
              Participant.resolve($0, directory: experts, webOrigin: context.server.authURL)
            },
            onAppear: { experts.ensureLoaded() })
        }
    }
  }
}

struct ChatsView: View {
  let context: SignedInContext
  let store: ChatListStore
  let experts: ExpertDirectory
  let open: (ConversationRoute) -> Void
  let showAccount: () -> Void

  var body: some View {
    content
      .navigationTitle("Chats")
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          Button(action: showAccount) {
            Image(systemName: "person.crop.circle")
          }
          .accessibilityLabel("Account")
        }
        ToolbarItem(placement: .topBarTrailing) {
          Menu {
            Section("New conversation with") {
              Button {
                open(.new(expertID: nil))
              } label: {
                Label("Otto", systemImage: "sparkles")
              }
              ForEach(experts.roster) { expert in
                Button {
                  open(.new(expertID: expert.id))
                } label: {
                  Label {
                    Text(expert.name)
                    if let title = expert.identity.jobTitle ?? expert.identity.role { Text(title) }
                  } icon: {
                    Image(systemName: "person.crop.circle")
                  }
                }
              }
            }
          } label: {
            Image(systemName: "square.and.pencil")
          }
          .accessibilityLabel("New conversation")
        }
      }
      .task { await store.refresh() }
  }

  @ViewBuilder
  private var content: some View {
    switch store.status {
    case .idle, .loading:
      ProgressView("Loading conversations")
    case .error:
      ContentUnavailableView {
        Label("Couldn't load conversations", systemImage: "exclamationmark.bubble")
      } description: {
        Text(store.error ?? "Couldn't load conversations.")
      } actions: {
        Button("Try Again") { Task { await store.refresh() } }
      }
    case .ready:
      if store.sessions.isEmpty {
        ScrollView {
          VStack(spacing: 24) {
            OttoArtwork(size: 144)
            VStack(spacing: 8) {
              Text("A little more off your plate.")
                .font(.title2.bold())
                .accessibilityAddTraits(.isHeader)
              Text(
                "Talk it through with Otto. Your conversations with Otto and your experts live here."
              )
              .font(.body)
              .foregroundStyle(.secondary)
            }
            .multilineTextAlignment(.center)
            Button("Message Otto") { open(.new(expertID: nil)) }
              .font(.headline)
              .buttonStyle(.borderedProminent)
              .buttonBorderShape(.capsule)
              .controlSize(.large)
              .accessibilityIdentifier("message-otto")
          }
          .frame(maxWidth: .infinity)
          .padding(.horizontal, 32)
          .padding(.vertical, 64)
        }
        .refreshable { await store.refresh() }
      } else {
        List {
          if let error = store.error {
            InlineNotice(text: error, tone: .danger).listRowSeparator(.hidden)
          }
          ForEach(store.sessions) { session in
            Button {
              open(.existing(session.id))
            } label: {
              ChatRow(
                session: session,
                participant: Participant.resolve(
                  session.expertID, directory: experts, webOrigin: context.server.authURL))
            }
            .accessibilityIdentifier("chat-row-\(session.id)")
            .onAppear {
              if session.id == store.sessions.last?.id { Task { await store.loadMore() } }
            }
          }
          if store.loadingMore {
            ProgressView().frame(maxWidth: .infinity).listRowSeparator(.hidden)
          }
        }
        .listStyle(.plain)
        .refreshable { await store.refresh() }
      }
    }
  }
}

private struct ChatRow: View {
  let session: SessionSummary
  let participant: Participant

  private var working: Bool {
    session.isProcessing || session.chatStatus == "running" || session.chatStatus == "queued"
  }

  private var status: String? {
    guard working else { return nil }
    return session.chatStatus == "queued" ? "Queued" : "Working…"
  }

  var body: some View {
    let title = session.title ?? "New conversation"
    let timestamp = ListTimestamp.format(session.updatedAt)
    HStack(spacing: 12) {
      AvatarView(
        name: participant.name, url: participant.avatarURL, size: 48, asset: participant.avatarAsset
      )
      VStack(alignment: .leading, spacing: 2) {
        HStack(spacing: 4) {
          Text(title).font(.headline).lineLimit(1)
          Spacer(minLength: 4)
          if session.isPinned {
            Image(systemName: "pin.fill").font(.caption2).foregroundStyle(.tertiary)
          }
          Text(timestamp).font(.subheadline).foregroundStyle(.secondary)
        }
        Text(status.map { "\(participant.name) · \($0)" } ?? participant.name)
          .font(.subheadline)
          .foregroundStyle(working ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
          .lineLimit(1)
      }
    }
    .padding(.vertical, 4)
    .contentShape(Rectangle())
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(
      [
        title, participant.name, status, timestamp.isEmpty ? nil : timestamp,
        session.isPinned ? "Pinned" : nil,
      ]
      .compactMap { $0 }.joined(separator: ", "))
  }
}

struct CompanyView: View {
  let context: SignedInContext
  let experts: ExpertDirectory
  let open: (ConversationRoute) -> Void
  let showAccount: () -> Void
  @State private var opening: String?
  @State private var openError: String?

  var body: some View {
    let otto = Participant.otto(webOrigin: context.server.authURL)
    List {
      if let openError {
        Section { InlineNotice(text: openError, tone: .danger) }
      }
      Section {
        Button {
          open(.new(expertID: nil))
        } label: {
          PersonRow(
            name: otto.name, subtitle: otto.role, avatarURL: otto.avatarURL,
            avatarAsset: otto.avatarAsset, busy: false)
        }
        .accessibilityHint("Starts a new conversation with Otto")
        .accessibilityIdentifier("company-otto")
      } header: {
        Text("Your Head of AI")
      } footer: {
        Text("The big picture, the next step, or a second opinion. Start with Otto.")
      }
      Section {
        if experts.roster.isEmpty {
          if experts.status == .ready {
            VStack(alignment: .leading, spacing: 8) {
              Image(systemName: "person.2")
                .font(.title2)
                .foregroundStyle(.secondary)
                .padding(.bottom, 8)
                .accessibilityHidden(true)
              Text("Room for your team")
                .font(.headline)
              Text(
                "The experts you hire with Otto will appear here, ready to pick up the conversation."
              )
              .font(.subheadline)
              .foregroundStyle(.secondary)
            }
            .padding(.vertical, 12)
          } else {
            Text(emptyRosterText).foregroundStyle(.secondary)
          }
        } else {
          ForEach(experts.roster) { expert in
            let participant = Participant.expert(
              expert.identity, expertID: expert.id, directoryReady: true,
              webOrigin: context.server.authURL)
            Button {
              openExpert(expert)
            } label: {
              PersonRow(
                name: expert.name,
                subtitle: [participant.role, expert.statusLine].compactMap { $0 }.joined(
                  separator: " · "),
                avatarURL: participant.avatarURL, busy: opening == expert.id)
            }
            .accessibilityHint("Opens your conversation with \(expert.name)")
            .accessibilityIdentifier("company-expert-\(expert.id)")
          }
        }
      } header: {
        Text("Your experts")
      } footer: {
        if !experts.roster.isEmpty {
          Text("Opens your most recent conversation with that expert.")
        }
      }
      if let error = experts.error, !experts.roster.isEmpty {
        Section { InlineNotice(text: error, tone: .danger) }
      }
    }
    .navigationTitle("Company")
    .toolbar {
      ToolbarItem(placement: .topBarLeading) {
        Button(action: showAccount) {
          Image(systemName: "person.crop.circle")
        }
        .accessibilityLabel("Account")
      }
    }
    .refreshable { await experts.load() }
    .task { await experts.load() }
  }

  private var emptyRosterText: String {
    switch experts.status {
    case .idle, .loading: "Loading your experts…"
    case .error: experts.error ?? "Couldn't load your experts."
    case .ready: "No experts hired yet."
    }
  }

  private func openExpert(_ expert: ExpertProfile) {
    guard opening == nil else { return }
    opening = expert.id
    openError = nil
    Task {
      do {
        let page = try await ChatAPI.sessions(
          context.api, limit: 1, offset: 0, expertID: expert.id, pinnedFirst: false)
        open(page.sessions.first.map { .existing($0.id) } ?? .new(expertID: expert.id))
      } catch {
        openError = ChatErrors.describe(error)
      }
      opening = nil
    }
  }
}

private struct PersonRow: View {
  let name: String
  let subtitle: String?
  let avatarURL: URL?
  var avatarAsset: String? = nil
  let busy: Bool

  var body: some View {
    HStack(spacing: 12) {
      AvatarView(name: name, url: avatarURL, size: 56, asset: avatarAsset)
      VStack(alignment: .leading, spacing: 2) {
        Text(name).font(.headline).foregroundStyle(.primary)
        if let subtitle, !subtitle.isEmpty {
          Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
        }
      }
      Spacer(minLength: 8)
      if busy {
        ProgressView()
      } else {
        Image(systemName: "chevron.right")
          .font(.footnote.weight(.semibold))
          .foregroundStyle(.tertiary)
          .accessibilityHidden(true)
      }
    }
    .padding(.vertical, 6)
    .contentShape(Rectangle())
  }
}
