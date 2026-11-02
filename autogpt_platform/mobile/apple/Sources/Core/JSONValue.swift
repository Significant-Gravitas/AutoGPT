import Foundation

/// Loosely typed JSON for untrusted server payloads. Unexpected shapes never
/// fail a whole response: accessors return nil and callers fall back.
enum JSONValue: Sendable, Hashable {
  case null
  case bool(Bool)
  case number(Double)
  case string(String)
  case array([JSONValue])
  case object([String: JSONValue])
}

extension JSONValue: Codable {
  init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
    } else if let value = try? container.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? container.decode(Double.self) {
      self = .number(value)
    } else if let value = try? container.decode(String.self) {
      self = .string(value)
    } else if let value = try? container.decode([JSONValue].self) {
      self = .array(value)
    } else if let value = try? container.decode([String: JSONValue].self) {
      self = .object(value)
    } else {
      throw DecodingError.dataCorruptedError(
        in: container, debugDescription: "Unsupported JSON value")
    }
  }

  func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .null: try container.encodeNil()
    case .bool(let value): try container.encode(value)
    case .number(let value): try container.encode(value)
    case .string(let value): try container.encode(value)
    case .array(let value): try container.encode(value)
    case .object(let value): try container.encode(value)
    }
  }
}

extension JSONValue {
  static func parse(_ data: Data) -> JSONValue? {
    guard !data.isEmpty else { return nil }
    return try? JSONDecoder().decode(JSONValue.self, from: data)
  }

  /// A JSON document, or the raw text when it isn't one.
  static func parseLoose(_ text: String) -> JSONValue {
    parse(Data(text.utf8)) ?? .string(text)
  }

  var object: [String: JSONValue]? {
    if case .object(let value) = self { return value }
    return nil
  }

  var array: [JSONValue]? {
    if case .array(let value) = self { return value }
    return nil
  }

  var string: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  var number: Double? {
    if case .number(let value) = self, value.isFinite { return value }
    return nil
  }

  var int: Int? {
    guard let number, number.rounded() == number else { return nil }
    return Int(exactly: number)
  }

  var bool: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }

  var isNull: Bool { self == .null }

  /// A trimmed string with content, otherwise nil.
  var nonEmptyString: String? {
    guard let string else { return nil }
    let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
  }

  subscript(key: String) -> JSONValue? { object?[key] }

  func encodedString(pretty: Bool = false) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
    guard let data = try? encoder.encode(self) else { return "" }
    return String(decoding: data, as: UTF8.self)
  }
}

extension Optional where Wrapped == JSONValue {
  /// JSON `null` and an absent field mean the same thing to every reader.
  var present: JSONValue? {
    switch self {
    case .none, .some(.null): return nil
    case .some(let value): return value
    }
  }
}
