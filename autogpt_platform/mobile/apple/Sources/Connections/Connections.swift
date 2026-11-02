import Foundation

/// A stored connection's public metadata. Secret values never reach the app.
struct ConnectionSummary: Identifiable, Equatable, Sendable {
  enum Kind: String, Sendable {
    case apiKey = "api_key"
    case oauth2
    case userPassword = "user_password"
    case hostScoped = "host_scoped"
    case other

    var label: String {
      switch self {
      case .apiKey: "API key"
      case .oauth2: "Signed-in account"
      case .userPassword: "Username and password"
      case .hostScoped: "Host-scoped credential"
      case .other: "Credential"
      }
    }
  }

  let id: String
  let provider: String
  let kind: Kind
  let title: String?
  let serviceName: String?
  let host: String?
  let username: String?
  let scopes: [String]
  let isManaged: Bool

  var displayName: String { title ?? serviceName ?? provider }
  var serviceLabel: String { serviceName ?? provider }

  init(
    id: String, provider: String, kind: Kind, title: String?, serviceName: String?, host: String?,
    username: String?, scopes: [String], isManaged: Bool
  ) {
    self.id = id
    self.provider = provider
    self.kind = kind
    self.title = title
    self.serviceName = serviceName
    self.host = host
    self.username = username
    self.scopes = scopes
    self.isManaged = isManaged
  }

  init?(_ raw: JSONValue) {
    guard let id = raw["id"]?.nonEmptyString, let provider = raw["provider"]?.nonEmptyString else {
      return nil
    }
    self.init(
      id: id, provider: provider, kind: raw["type"]?.string.flatMap(Kind.init) ?? .other,
      title: raw["title"]?.nonEmptyString, serviceName: raw["service_name"]?.nonEmptyString,
      host: raw["host"]?.nonEmptyString, username: raw["username"]?.nonEmptyString,
      scopes: (raw["scopes"]?.array ?? []).compactMap(\.string),
      isManaged: raw["is_managed"]?.bool == true)
  }
}

/// An edit keeps the connection's identity, so workflows and experts that
/// reference it keep working. A nil secret means unchanged; secrets are write-only.
struct ConnectionChange: Equatable, Sendable {
  let title: String
  let replacementSecret: String?
}

enum ConnectionEditOutcome: Equatable, Sendable {
  case saved(ConnectionSummary)
  /// The provider or server refused the new value; nothing changed.
  case rejected(String)
  /// Someone changed the connection elsewhere since it was loaded.
  case conflict
}

protocol ConnectionService: Sendable {
  /// Whether this server can edit an existing connection in place.
  var canEdit: Bool { get }
  func list() async throws -> [ConnectionSummary]
  func update(_ id: String, _ change: ConnectionChange) async throws -> ConnectionEditOutcome
}

struct ConnectionEditingUnavailable: UserFacingError {
  var userMessage: String { "This server can't edit connections from the app yet." }
}

/// The production server lists connections today. Editing in place is an
/// identified server gap (no update route that preserves the credential's
/// identity), so this service reports it can't edit rather than offering a
/// control that would fail or replace the credential.
struct LiveConnectionService: ConnectionService {
  let api: BackendClient

  var canEdit: Bool { false }

  func list() async throws -> [ConnectionSummary] {
    (try await api.json("/api/integrations/credentials")?.array ?? []).compactMap(
      ConnectionSummary.init)
  }

  func update(_ id: String, _ change: ConnectionChange) async throws -> ConnectionEditOutcome {
    throw ConnectionEditingUnavailable()
  }
}
