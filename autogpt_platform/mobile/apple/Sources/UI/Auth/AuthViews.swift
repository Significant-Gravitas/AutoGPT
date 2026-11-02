import SwiftUI

enum AuthRoute: Hashable {
  case ownServer, signIn
}

struct AuthFlowView: View {
  let session: SessionStore
  @State private var path: [AuthRoute] = []

  var body: some View {
    NavigationStack(path: $path) {
      WelcomeView(session: session, path: $path)
        .navigationDestination(for: AuthRoute.self) { route in
          switch route {
          case .ownServer: OwnServerView(session: session, path: $path)
          case .signIn: SignInView(session: session, path: $path)
          }
        }
    }
  }
}

struct WelcomeView: View {
  let session: SessionStore
  @Binding var path: [AuthRoute]
  @State private var forgetting: ServerConfig?

  var body: some View {
    List {
      Section {
        VStack(spacing: 16) {
          OttoArtwork(size: 160)
          Text("AutoGPT")
            .font(.largeTitle.bold())
            .accessibilityAddTraits(.isHeader)
          Text("Your AI company,\none conversation away.")
            .font(.title3)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
        .listRowBackground(Color.clear)
      }
      if let notice = session.signedOutNotice {
        Section { InlineNotice(text: notice, tone: .danger) }
      }
      Section {
        VStack(spacing: 12) {
          Button {
            session.chooseServer(.cloud)
            path.append(.signIn)
          } label: {
            Text("Continue with AutoGPT Cloud")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.borderedProminent)
          .accessibilityIdentifier("choose-cloud")
          Button {
            path.append(.ownServer)
          } label: {
            Text("Use your own server")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.bordered)
          .accessibilityIdentifier("choose-own-server")
        }
        .font(.headline)
        .buttonBorderShape(.capsule)
        .controlSize(.large)
        .listRowBackground(Color.clear)
        .listRowInsets(EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 0))
      } footer: {
        Text("Sign in to your AutoGPT account to get started.")
          .frame(maxWidth: .infinity)
          .multilineTextAlignment(.center)
      }
      if !session.customServers.isEmpty {
        Section {
          ForEach(session.customServers) { server in
            Button {
              session.chooseServer(server)
              path.append(.signIn)
            } label: {
              ServerRow(title: server.name, subtitle: server.authHost, symbol: "network")
            }
            .contextMenu {
              Button("Forget Server", systemImage: "trash", role: .destructive) {
                forgetting = server
              }
            }
            .swipeActions {
              Button("Forget", role: .destructive) { forgetting = server }
            }
            .accessibilityAction(named: "Forget server") { forgetting = server }
          }
        } header: {
          Text("Recent servers")
        } footer: {
          Text("Touch and hold a server to forget it.")
        }
      }
    }
    .scrollContentBackground(.hidden)
    .background(Color(uiColor: .systemBackground))
    .toolbar(.hidden, for: .navigationBar)
    .alert(
      "Forget \(forgetting?.name ?? "server")?",
      isPresented: Binding(get: { forgetting != nil }, set: { if !$0 { forgetting = nil } }),
      presenting: forgetting
    ) { server in
      Button("Cancel", role: .cancel) { forgetting = nil }
      Button("Forget", role: .destructive) {
        session.forgetServer(server.id)
        forgetting = nil
      }
    } message: { _ in
      Text("You can add it again later.")
    }
  }
}

private struct ServerRow: View {
  let title: String
  let subtitle: String
  let symbol: String

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: symbol)
        .font(.title3)
        .foregroundStyle(.tint)
        .frame(width: 30)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 2) {
        Text(title).foregroundStyle(.primary)
        Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
      }
      Spacer(minLength: 8)
      Image(systemName: "chevron.right")
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.tertiary)
        .accessibilityHidden(true)
    }
    .contentShape(Rectangle())
  }
}

struct OwnServerView: View {
  let session: SessionStore
  @Binding var path: [AuthRoute]
  @State private var name = ""
  @State private var authInput = ""
  @State private var apiInput = ""
  @State private var authError: String?
  @State private var apiError: String?
  @FocusState private var focus: Field?

  private enum Field { case auth, api, name }

  var body: some View {
    Form {
      Section {
        FieldRow(label: "Sign-in address") {
          TextField(
            "Sign-in address", text: $authInput,
            prompt: Text(verbatim: "https://autogpt.example.com")
          )
          .urlEntry()
          .focused($focus, equals: .auth)
          .submitLabel(.next)
          .onSubmit { focus = .api }
          .accessibilityIdentifier("own-server-auth-url")
        }
        FieldRow(label: "API address") {
          TextField(
            "API address", text: $apiInput,
            prompt: Text(verbatim: "https://api.autogpt.example.com")
          )
          .urlEntry()
          .focused($focus, equals: .api)
          .submitLabel(.next)
          .onSubmit { focus = .name }
          .accessibilityIdentifier("own-server-api-url")
        }
        FieldRow(label: "Name (optional)") {
          TextField("Name (optional)", text: $name, prompt: Text("Team server"))
            .focused($focus, equals: .name)
            .submitLabel(.done)
            .onSubmit(submit)
        }
      } header: {
        Text("Server")
      } footer: {
        Text(
          "Enter your AutoGPT web address, which handles sign-in, and its API address. Plain http:// only works for localhost."
        )
      }
      if authError != nil || apiError != nil {
        Section {
          if let authError {
            Text(verbatim: "Sign-in address: \(authError)").foregroundStyle(.red)
          }
          if let apiError {
            Text(verbatim: "API address: \(apiError)").foregroundStyle(.red)
          }
        }
      }
      Section {
        Button(action: submit) {
          Text("Continue").bold().frame(maxWidth: .infinity)
        }
      } footer: {
        Text(
          "Your server's addresses are checked when you sign in. Your sign-in is only ever sent to the sign-in address."
        )
      }
    }
    .navigationTitle("Your Server")
    .navigationBarTitleDisplayMode(.inline)
  }

  private func submit() {
    let auth = Origin.normalize(authInput)
    let api = Origin.normalize(apiInput)
    authError = auth.failure
    apiError = api.failure
    guard case .valid(let authURL) = auth, case .valid(let apiURL) = api else {
      let messages = [
        authError.map { "Sign-in address: \($0)" }, apiError.map { "API address: \($0)" },
      ]
      AccessibilityNotification.Announcement(messages.compactMap { $0 }.joined(separator: " "))
        .post()
      return
    }
    focus = nil
    session.rememberServer(.custom(name: name, authURL: authURL, apiURL: apiURL))
    path.append(.signIn)
  }
}

struct SignInView: View {
  let session: SessionStore
  @Binding var path: [AuthRoute]
  @State private var email = ""
  @State private var password = ""
  @State private var error: String?
  @State private var busy = false
  @State private var failures = 0
  @FocusState private var focus: Field?

  private enum Field { case email, password }

  var body: some View {
    Group {
      if let server = session.signedOutServer {
        form(server)
      } else {
        ContentUnavailableView {
          Label("Choose a server", systemImage: "server.rack")
        } description: {
          Text("Pick AutoGPT Cloud or your own server before signing in.")
        } actions: {
          Button("Choose Server") { path.removeAll() }
        }
      }
    }
    .navigationTitle("Sign In")
    .navigationBarTitleDisplayMode(.inline)
    .sensoryFeedback(.error, trigger: failures)
  }

  private func form(_ server: ServerConfig) -> some View {
    Form {
      Section("Server") {
        HStack(spacing: 12) {
          Image(systemName: server.kind == .cloud ? "cloud" : "server.rack")
            .font(.title3)
            .foregroundStyle(.tint)
            .frame(width: 30)
            .accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 2) {
            Text(server.name)
            Text(server.authHost).font(.subheadline).foregroundStyle(.secondary)
          }
          Spacer(minLength: 8)
          Button("Change") {
            if !path.isEmpty { path.removeLast() }
          }
          .buttonStyle(.borderless)
        }
      }
      Section("Account") {
        FieldRow(label: "Email") {
          TextField("Email", text: $email, prompt: Text(verbatim: "you@example.com"))
            .textContentType(.username)
            .keyboardType(.emailAddress)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .submitLabel(.next)
            .focused($focus, equals: .email)
            .onSubmit { focus = .password }
            .disabled(busy)
            .accessibilityIdentifier("sign-in-email")
        }
        FieldRow(label: "Password") {
          SecureField("Password", text: $password, prompt: Text("Required"))
            .textContentType(.password)
            .submitLabel(.go)
            .focused($focus, equals: .password)
            .onSubmit(submit)
            .disabled(busy)
            .accessibilityIdentifier("sign-in-password")
        }
      }
      if let error {
        Section {
          Text(error)
            .foregroundStyle(.red)
            .accessibilityIdentifier("sign-in-error")
        }
      }
      Section {
        Button(action: submit) {
          HStack {
            Spacer()
            if busy {
              ProgressView()
            } else {
              Text("Sign In").bold()
            }
            Spacer()
          }
        }
        .disabled(busy)
        .accessibilityLabel(busy ? "Signing In…" : "Sign In")
        .accessibilityIdentifier("sign-in-submit")
      } footer: {
        VStack(alignment: .leading, spacing: 8) {
          if let note = GoogleSignIn.readiness(for: server).explanation { Text(note) }
          Text("New to AutoGPT? Create your account on the web, then sign in here.")
        }
      }
    }
  }

  private func submit() {
    guard !busy else { return }
    guard !email.trimmed.isEmpty, !password.isEmpty else {
      error = "Enter your email and password."
      return
    }
    busy = true
    error = nil
    focus = nil
    Task {
      do {
        try await session.signIn(email: email, password: password)
      } catch let failure as SignInError {
        error = failure.message
        failures += 1
        if failure.kind == .credentials { password = "" }
        AccessibilityNotification.Announcement(failure.message).post()
      } catch {
        self.error = "Couldn't sign in. Try again."
        failures += 1
      }
      busy = false
    }
  }
}

extension View {
  fileprivate func urlEntry() -> some View {
    keyboardType(.URL)
      .textContentType(.URL)
      .textInputAutocapitalization(.never)
      .autocorrectionDisabled()
  }
}
