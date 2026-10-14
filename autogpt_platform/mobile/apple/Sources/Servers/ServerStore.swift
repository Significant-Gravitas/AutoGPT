import Foundation

/// Which server is chosen, and which of its accounts (by partition key) is signed in.
struct ServerSelection: Codable, Equatable, Sendable {
  let serverID: String
  let accountKey: String?
}

/// Non-secret app state lives in the app container, which iOS deletes on
/// uninstall (unlike the Keychain). Credentials never go here.
enum ServerStore {
  private static let serversKey = "servers.v1"
  private static let selectionKey = "selection.v1"
  private static let maxRemembered = 10

  static func customServers() -> [ServerConfig] {
    guard let data = UserDefaults.standard.data(forKey: serversKey),
      let servers = try? JSONDecoder().decode([ServerConfig].self, from: data)
    else { return [] }
    return servers.filter { $0.kind == .custom && $0.isValid }
  }

  @discardableResult
  static func remember(_ server: ServerConfig) -> [ServerConfig] {
    let next = Array(
      ([server] + customServers().filter { $0.id != server.id }).prefix(maxRemembered))
    write(next)
    return next
  }

  @discardableResult
  static func forget(_ serverID: String) -> [ServerConfig] {
    let next = customServers().filter { $0.id != serverID }
    write(next)
    return next
  }

  static func server(id: String) -> ServerConfig? {
    id == ServerConfig.cloud.id ? .cloud : customServers().first { $0.id == id }
  }

  static func selection() -> ServerSelection? {
    guard let data = UserDefaults.standard.data(forKey: selectionKey) else { return nil }
    return try? JSONDecoder().decode(ServerSelection.self, from: data)
  }

  static func saveSelection(_ selection: ServerSelection) {
    if let data = try? JSONEncoder().encode(selection) {
      UserDefaults.standard.set(data, forKey: selectionKey)
    }
  }

  private static func write(_ servers: [ServerConfig]) {
    if let data = try? JSONEncoder().encode(servers) {
      UserDefaults.standard.set(data, forKey: serversKey)
    }
  }
}
