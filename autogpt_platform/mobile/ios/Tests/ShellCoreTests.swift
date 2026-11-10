import XCTest

@testable import ShellCore

final class ShellCoreTests: XCTestCase {
  func testOnlyTheExactHTTPSOriginIsTrusted() {
    let policy = NavigationPolicy(origin: URL(string: "https://platform.agpt.co")!)
    XCTAssertTrue(policy.isTrusted(URL(string: "https://platform.agpt.co/home?sessionId=1")!))
    for value in [
      "https://platform.agpt.co.evil.com", "http://platform.agpt.co",
      "https://platform.agpt.co:444", "https://evil.com@platform.agpt.co", "file:///etc/passwd",
      "javascript:alert(1)",
    ] {
      XCTAssertFalse(policy.isTrusted(URL(string: value)!), value)
    }
  }

  func testCallbackNeedsTheRightStateAndAnExactRoute() throws {
    let attempt = AuthAttempt()
    let code = String(repeating: "a", count: 43)
    let valid = URL(string: "autogpt://auth/callback?code=\(code)&state=\(attempt.state)")!
    XCTAssertEqual(
      attempt.exchangeBody(callback: valid), "code=\(code)&verifier=\(attempt.verifier)")
    for value in [
      "autogpt://auth/callback?code=\(code)&state=wrong",
      "autogpt://evil/callback?code=\(code)&state=\(attempt.state)",
      "https://auth/callback?code=\(code)&state=\(attempt.state)",
    ] {
      XCTAssertNil(attempt.exchangeBody(callback: URL(string: value)!))
    }
    XCTAssertNil(
      attempt.exchangeBody(callback: URL(string: valid.absoluteString + "&code=\(code)")!))
  }

  func testAnExpiredAttemptCannotRedeemItsCallback() {
    let attempt = AuthAttempt(createdAt: Date(timeIntervalSinceNow: -600))
    let callback = URL(
      string:
        "autogpt://auth/callback?code=\(String(repeating: "a", count: 43))&state=\(attempt.state)")!
    XCTAssertNil(attempt.exchangeBody(callback: callback))
  }

  func testBlobDownloadsMustBelongToTheTrustedOrigin() {
    let policy = NavigationPolicy(origin: URL(string: "https://platform.agpt.co")!)
    XCTAssertTrue(policy.isTrustedBlob(URL(string: "blob:https://platform.agpt.co/file")!))
    for value in [
      "blob:https://evil.example.com/file", "blob:http://platform.agpt.co/file",
      "blob:https://platform.agpt.co:444/file", "blob:null/file",
      "blob:https://evil.com@platform.agpt.co/file", "https://platform.agpt.co/file",
    ] {
      XCTAssertFalse(policy.isTrustedBlob(URL(string: value)!), value)
    }
  }
}
