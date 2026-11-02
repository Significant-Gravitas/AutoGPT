import SwiftUI

struct AccountView: View {
  let context: SignedInContext
  let session: SessionStore
  @Environment(\.dismiss) private var dismiss
  @State private var confirmingSignOut = false
  @State private var signingOut = false
  @State private var notificationPermission: String?

  var body: some View {
    let account = context.account
    let server = context.server
    NavigationStack {
      List {
        Section {
          HStack(spacing: 12) {
            AvatarView(name: account.name ?? account.email, url: nil, size: 48)
            VStack(alignment: .leading, spacing: 2) {
              Text(account.name ?? account.email).font(.headline)
              if account.name != nil {
                Text(account.email).font(.subheadline).foregroundStyle(.secondary)
              }
            }
          }
          .accessibilityElement(children: .combine)
        }
        Section {
          LabeledContent(
            server.name, value: server.kind == .cloud ? "Hosted by AutoGPT" : "Your own server")
          LabeledContent("Sign-in", value: server.authHost)
          LabeledContent("API", value: server.apiHost)
        } header: {
          Text("Server")
        } footer: {
          Text(
            "Your sign-in only goes to this server's sign-in address, and is kept in this device's keychain."
          )
        }
        Section {
          NavigationLink {
            ConnectionsListView(service: LiveConnectionService(api: context.api))
          } label: {
            Label("Connections", systemImage: "key")
          }
          LabeledContent {
            Text("Not in the app yet")
          } label: {
            Label("Subscription", systemImage: "creditcard")
          }
          LabeledContent {
            Text(notificationPermission ?? "Checking…")
          } label: {
            Label("Notifications", systemImage: "bell")
          }
        } footer: {
          VStack(alignment: .leading, spacing: 6) {
            if let reason = Purchases.readiness.explanation { Text(reason) }
            if let reason = PushNotifications.readiness.explanation { Text(reason) }
          }
        }
        Section {
          Button(role: .destructive) {
            confirmingSignOut = true
          } label: {
            Text(signingOut ? "Signing Out…" : "Sign Out")
          }
          .disabled(signingOut)
          .accessibilityIdentifier("sign-out")
        } footer: {
          Text("To use a different account or server, sign out first.")
        }
      }
      .navigationTitle("Account")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") { dismiss() }
        }
      }
      .alert("Sign out?", isPresented: $confirmingSignOut) {
        Button("Cancel", role: .cancel) {}
        Button("Sign Out", role: .destructive) {
          signingOut = true
          Task { await session.signOut() }
        }
      } message: {
        Text("This removes \(account.email)'s sign-in from this device.")
      }
      .task { notificationPermission = await PushNotifications.systemPermission() }
    }
  }
}

/// Connections on the selected server. Editing is offered only when the
/// service can change a connection in place; otherwise each row is read-only
/// and the footer says why.
struct ConnectionsListView: View {
  let service: any ConnectionService
  @State private var connections: [ConnectionSummary] = []
  @State private var loading = true
  @State private var error: String?

  var body: some View {
    List {
      if let error {
        Section { InlineNotice(text: error, tone: .danger) }
      }
      Section {
        if connections.isEmpty, !loading, error == nil {
          Text("No connections yet.").foregroundStyle(.secondary)
        }
        ForEach(connections) { connection in
          if service.canEdit {
            NavigationLink {
              ConnectionEditorView(connection: connection, service: service) { updated in
                if let index = connections.firstIndex(where: { $0.id == updated.id }) {
                  connections[index] = updated
                }
              }
            } label: {
              ConnectionRow(connection: connection)
            }
            .accessibilityIdentifier("connection-row-\(connection.id)")
          } else {
            ConnectionRow(connection: connection)
              .accessibilityIdentifier("connection-row-\(connection.id)")
          }
        }
      } footer: {
        if !service.canEdit {
          Text(
            "Adding, editing and reconnecting services in the app needs a server update that isn't available yet. Manage connections on the web for now."
          )
        }
      }
    }
    .overlay {
      if loading && connections.isEmpty { ProgressView() }
    }
    .navigationTitle("Connections")
    .refreshable { await load() }
    .task { await load() }
  }

  private func load() async {
    loading = true
    do {
      connections = try await service.list()
      error = nil
    } catch {
      self.error = errorMessage(error, fallback: "Couldn't load your connections.")
    }
    loading = false
  }
}

private struct ConnectionRow: View {
  let connection: ConnectionSummary

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(connection.displayName)
      Text(
        [connection.serviceLabel, connection.kind.label, connection.host].compactMap { $0 }.joined(
          separator: " · ")
      )
      .font(.subheadline)
      .foregroundStyle(.secondary)
    }
    .accessibilityElement(children: .combine)
  }
}

/// Edits one connection in place. The secret is write-only: the current value
/// is never shown, leaving it untouched keeps it, and an empty replacement is
/// refused. A host-scoped destination can't be changed here.
struct ConnectionEditorView: View {
  let connection: ConnectionSummary
  let service: any ConnectionService
  let onSaved: (ConnectionSummary) -> Void
  @Environment(\.dismiss) private var dismiss
  @State private var title: String
  @State private var replacingSecret = false
  @State private var secret = ""
  @State private var saving = false
  @State private var error: String?
  @State private var saves = 0

  init(
    connection: ConnectionSummary, service: any ConnectionService,
    onSaved: @escaping (ConnectionSummary) -> Void
  ) {
    self.connection = connection
    self.service = service
    self.onSaved = onSaved
    _title = State(initialValue: connection.title ?? "")
  }

  private var secretLabel: String {
    connection.kind == .userPassword ? "Password" : "API key"
  }

  private var canReplaceSecret: Bool {
    [.apiKey, .userPassword, .hostScoped].contains(connection.kind)
  }

  private var changed: Bool {
    title.trimmed != (connection.title ?? "") || replacingSecret
  }

  var body: some View {
    Form {
      Section {
        FieldRow(label: "Name") {
          TextField("Name", text: $title, prompt: Text(connection.serviceLabel))
            .disabled(saving)
            .accessibilityIdentifier("connection-name")
        }
        LabeledContent("Service", value: connection.serviceLabel)
        LabeledContent("Type", value: connection.kind.label)
        if let host = connection.host {
          LabeledContent("Host", value: host)
        }
      } footer: {
        if connection.host != nil {
          Text(
            "A host-scoped connection's destination can't be changed here. Add a new connection for another host."
          )
        }
      }
      if canReplaceSecret {
        Section {
          Toggle("Replace \(secretLabel.lowercased())", isOn: $replacingSecret.animation())
            .disabled(saving)
            .accessibilityIdentifier("connection-replace-secret")
          if replacingSecret {
            FieldRow(label: "New \(secretLabel.lowercased())") {
              SecureField(
                "New \(secretLabel.lowercased())", text: $secret, prompt: Text("Required")
              )
              .textContentType(.password)
              .disabled(saving)
              .accessibilityIdentifier("connection-secret")
            }
          }
        } footer: {
          Text(
            "The current \(secretLabel.lowercased()) is never shown. Leave it unchanged to keep it."
          )
        }
      } else if connection.kind == .oauth2 {
        Section {
          LabeledContent("Account", value: connection.username ?? "Signed in")
          if !connection.scopes.isEmpty {
            LabeledContent("Access", value: "\(connection.scopes.count) permissions")
          }
        } footer: {
          Text(
            "Reconnecting this account opens the provider in the system browser. That handoff isn't available in the app yet."
          )
        }
      }
      if let error {
        Section {
          Text(error).foregroundStyle(.red).accessibilityIdentifier("connection-error")
        }
      }
    }
    .navigationTitle(connection.displayName)
    .navigationBarTitleDisplayMode(.inline)
    .toolbar {
      ToolbarItem(placement: .confirmationAction) {
        if saving {
          ProgressView()
        } else {
          Button("Save", action: save)
            .disabled(!changed)
            .accessibilityIdentifier("connection-save")
        }
      }
    }
    .sensoryFeedback(.success, trigger: saves)
  }

  private func save() {
    if title.trimmed.isEmpty {
      error = "Give this connection a name."
      return
    }
    if replacingSecret && secret.trimmed.isEmpty {
      error = "Enter the new \(secretLabel.lowercased()), or turn off Replace."
      return
    }
    error = nil
    saving = true
    let change = ConnectionChange(
      title: title.trimmed, replacementSecret: replacingSecret ? secret : nil)
    Task {
      do {
        switch try await service.update(connection.id, change) {
        case .saved(let updated):
          secret = ""
          saves += 1
          onSaved(updated)
          dismiss()
        case .rejected(let message):
          error = message
        case .conflict:
          error = "This connection changed somewhere else. Go back, reload it and try again."
        }
      } catch {
        self.error = errorMessage(error, fallback: "Couldn't save this connection. Try again.")
      }
      saving = false
    }
  }
}
