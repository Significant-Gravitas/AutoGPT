import CryptoKit
import Foundation

enum Digest {
  /// A short, Keychain-safe digest used to partition stored data.
  static func stable(_ value: String) -> String {
    let hex = SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
    return String(hex.prefix(32))
  }
}

struct DeadlineExceeded: Error {}

/// Runs `operation` and fails with `DeadlineExceeded` if it hasn't finished in
/// time. The losing child is cancelled, which cancels any network request.
func withDeadline<T: Sendable>(
  _ duration: Duration,
  operation: @escaping @Sendable () async throws -> T
) async throws -> T {
  try await withThrowingTaskGroup(of: T.self) { group in
    group.addTask { try await operation() }
    group.addTask {
      try await Task.sleep(for: duration)
      throw DeadlineExceeded()
    }
    defer { group.cancelAll() }
    guard let result = try await group.next() else { throw CancellationError() }
    return result
  }
}

/// Sleeps that a newer event can cut short, such as a backoff the user
/// skips by pulling to refresh.
@MainActor
final class Waker {
  private var sleepers: [UUID: Task<Void, Never>] = [:]

  func pause(_ duration: Duration) async {
    guard duration > .zero else { return }
    let id = UUID()
    let sleeper = Task { _ = try? await Task.sleep(for: duration) }
    sleepers[id] = sleeper
    await withTaskCancellationHandler {
      await sleeper.value
    } onCancel: {
      sleeper.cancel()
    }
    sleepers[id] = nil
  }

  func rouse() {
    for sleeper in sleepers.values { sleeper.cancel() }
    sleepers.removeAll()
  }
}

enum ISODate {
  nonisolated(unsafe) private static let fractional: ISO8601DateFormatter = {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter
  }()

  nonisolated(unsafe) private static let whole: ISO8601DateFormatter = {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter
  }()

  private static let fraction = try! NSRegularExpression(
    pattern: #"\.(\d+)"#)
  private static let zone = try! NSRegularExpression(
    pattern: #"(Z|[+-]\d{2}:?\d{2})$"#)

  /// Python writes microseconds and sometimes a naive UTC time; the system
  /// parser wants milliseconds and an explicit zone.
  static func parse(_ string: String) -> Date? {
    var text = string.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { return nil }
    let whole = NSRange(text.startIndex..., in: text)
    if let match = fraction.firstMatch(in: text, range: whole),
      let digitsRange = Range(match.range(at: 1), in: text),
      let fullRange = Range(match.range, in: text)
    {
      let digits = String(text[digitsRange].prefix(3))
      let padded = digits.padding(toLength: 3, withPad: "0", startingAt: 0)
      text.replaceSubrange(fullRange, with: ".\(padded)")
    }
    let afterTime = NSRange(text.startIndex..., in: text)
    if zone.firstMatch(in: text, range: afterTime) == nil { text += "Z" }
    return Self.fractional.date(from: text) ?? Self.whole.date(from: text)
  }
}

enum ListTimestamp {
  /// Compact list timestamps in the style of system Messages: time, weekday, or date.
  static func format(_ iso: String?, now: Date = Date()) -> String {
    guard let iso, let date = ISODate.parse(iso) else { return "" }
    let elapsed = now.timeIntervalSince(date)
    if elapsed < 60 { return "Now" }
    if elapsed < 86_400, Calendar.current.isDate(date, inSameDayAs: now) {
      return date.formatted(date: .omitted, time: .shortened)
    }
    if elapsed < 7 * 86_400 {
      return date.formatted(.dateTime.weekday(.abbreviated))
    }
    return date.formatted(.dateTime.month(.abbreviated).day())
  }
}

extension String {
  var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }

  func truncated(to max: Int) -> String {
    count > max ? String(prefix(max - 1)) + "…" : self
  }

  /// Percent-encodes everything outside RFC 3986's unreserved set, for
  /// query values and path segments alike.
  var strictlyPercentEncoded: String {
    addingPercentEncoding(withAllowedCharacters: .strictlyUnreserved) ?? ""
  }
}

extension CharacterSet {
  static let strictlyUnreserved = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
}
