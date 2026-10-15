import Foundation

/// The lightweight identity the chat surfaces need, including let-go experts.
struct ExpertIdentity: Identifiable, Equatable, Sendable {
  let id: String
  let name: String
  let avatarPath: String?
  let role: String?
  let jobTitle: String?
  let isArchived: Bool

  init?(_ raw: JSONValue) {
    guard let id = raw["id"]?.nonEmptyString, let name = raw["name"]?.nonEmptyString else {
      return nil
    }
    self.id = id
    self.name = name
    avatarPath = raw["avatar_url"]?.nonEmptyString
    role = raw["role"]?.nonEmptyString
    jobTitle = raw["job_title"]?.nonEmptyString
    isArchived = raw["is_archived"]?.bool == true
  }
}

struct ExpertProfile: Identifiable, Equatable, Sendable {
  let identity: ExpertIdentity
  let tagline: String?
  let lastRunAt: String?
  let lastRunStatus: String?
  let setupStatus: String?
  let schedulesPausedAt: String?

  var id: String { identity.id }
  var name: String { identity.name }

  init?(_ raw: JSONValue) {
    guard let identity = ExpertIdentity(raw) else { return nil }
    self.identity = identity
    tagline = raw["tagline"]?.nonEmptyString
    lastRunAt = raw["last_run_at"]?.string
    lastRunStatus = raw["last_run_status"]?.nonEmptyString
    setupStatus = raw["setup_status"]?.nonEmptyString
    schedulesPausedAt = raw["schedules_paused_at"]?.string
  }

  /// Setup, paused schedules or the last run, whichever matters most.
  var statusLine: String? {
    if let setupStatus, setupStatus != "ready" { return "Setting up" }
    if schedulesPausedAt != nil { return "Schedules paused" }
    guard let lastRunAt else { return nil }
    let when = ListTimestamp.format(lastRunAt)
    return lastRunStatus.map { "Last run \(when), \($0.lowercased())" } ?? "Last run \(when)"
  }
}

enum ExpertsAPI {
  static func identities(_ api: BackendClient) async throws -> [ExpertIdentity] {
    (try await api.json("/api/experts/identities")?.array ?? []).compactMap(ExpertIdentity.init)
  }

  /// The active hired roster with status fields for the Company screen.
  static func roster(_ api: BackendClient) async throws -> [ExpertProfile] {
    (try await api.json("/api/experts")?.array ?? []).compactMap(ExpertProfile.init)
      .filter { !$0.identity.isArchived }
  }
}
