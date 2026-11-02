import Foundation
import Security

/// A signed Better Auth session credential, bound to the origins it was
/// issued for. It is only ever sent to `authOrigin`.
struct StoredCredential: Codable, Equatable, Sendable {
  var version = 1
  var sessionToken: String
  let authOrigin: String
  let apiOrigin: String
  let userID: String
  let email: String
  let name: String?
  var savedAt: Date

  var isUsable: Bool { version == 1 && sessionToken.contains(".") }
}

struct KeychainError: Error {
  let status: OSStatus
}

/// Keychain items partitioned by server and account, readable only while the
/// device is unlocked and never migrated to another device.
enum CredentialStore {
  private static let service = "com.agpt.mobile.experimental.session"
  private static let installMarker = "installation.v1"

  static func key(serverPartition: String, accountKey: String) -> String {
    "autogpt.credential.v1.\(serverPartition).\(accountKey)"
  }

  static func read(_ key: String) throws -> StoredCredential? {
    var query = baseQuery(key)
    query[kSecReturnData as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var item: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &item)
    if status == errSecItemNotFound { return nil }
    guard status == errSecSuccess else { throw KeychainError(status: status) }
    guard let data = item as? Data,
      let credential = try? JSONDecoder().decode(StoredCredential.self, from: data),
      credential.isUsable
    else { return nil }
    return credential
  }

  static func write(_ key: String, _ credential: StoredCredential) throws {
    let data = try JSONEncoder().encode(credential)
    let update =
      [
        kSecValueData as String: data,
        kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
      ] as [String: Any]
    var status = SecItemUpdate(baseQuery(key) as CFDictionary, update as CFDictionary)
    if status == errSecItemNotFound {
      var add = baseQuery(key)
      add.merge(update) { _, new in new }
      status = SecItemAdd(add as CFDictionary, nil)
    }
    guard status == errSecSuccess else { throw KeychainError(status: status) }
  }

  static func delete(_ key: String) {
    SecItemDelete(baseQuery(key) as CFDictionary)
  }

  /// iOS keeps Keychain items after the app is deleted. The install marker
  /// lives in the app container, so its absence means a fresh install: any
  /// session a previous install left behind is removed before anything can use it.
  static func wipeLeftoversFromPreviousInstall() {
    let defaults = UserDefaults.standard
    guard !defaults.bool(forKey: installMarker) else { return }
    let everything =
      [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service,
      ] as [String: Any]
    SecItemDelete(everything as CFDictionary)
    defaults.set(true, forKey: installMarker)
  }

  private static func baseQuery(_ key: String) -> [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: key,
    ]
  }
}
