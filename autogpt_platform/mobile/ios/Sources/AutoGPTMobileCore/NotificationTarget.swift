import Foundation

public enum NotificationTarget {
  public static func url(
    path: String, notificationOrigin: String, binding: String, expectedBinding: String,
    origin: AppOrigin
  ) -> URL? {
    guard !binding.isEmpty, binding == expectedBinding,
      notificationOrigin == origin.url.absoluteString,
      path.hasPrefix("/"), !path.hasPrefix("//"),
      let parts = URLComponents(string: path), parts.scheme == nil, parts.host == nil,
      parts.fragment == nil, let items = parts.queryItems, items.count == 1,
      let item = items.first, let value = item.value, !value.isEmpty, value.count <= 128
    else { return nil }
    guard
      (parts.path == "/home" && item.name == "sessionId")
        || (parts.path == "/mobile" && item.name == "tab" && value == "attention")
    else { return nil }
    guard let url = URL(string: path, relativeTo: origin.url)?.absoluteURL, origin.contains(url)
    else { return nil }
    return url
  }
}
