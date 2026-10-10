import Foundation
import Testing

@testable import AutoGPTMobileCore

@Test func notificationTargetsStayInsideTheCurrentAccountAndServer() throws {
  let origin = try AppOrigin("https://platform.agpt.co")
  #expect(
    NotificationTarget.url(
      path: "/home?sessionId=hello", notificationOrigin: origin.url.absoluteString,
      binding: "current", expectedBinding: "current", origin: origin)?.path == "/home")
  for path in [
    "https://attacker.example", "//attacker.example/home", "/api/auth/sign-out",
    "/home?next=https://attacker.example", "/home?sessionId=one&sessionId=two",
  ] {
    #expect(
      NotificationTarget.url(
        path: path, notificationOrigin: origin.url.absoluteString, binding: "current",
        expectedBinding: "current", origin: origin) == nil)
  }
  #expect(
    NotificationTarget.url(
      path: "/home?sessionId=hello", notificationOrigin: origin.url.absoluteString,
      binding: "old-account", expectedBinding: "current", origin: origin) == nil)
  #expect(
    NotificationTarget.url(
      path: "/mobile?tab=attention", notificationOrigin: "https://other.example",
      binding: "current", expectedBinding: "current", origin: origin) == nil)
}
