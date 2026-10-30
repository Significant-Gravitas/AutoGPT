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

  func testCallbackNeedsTheRightStateAndOurHTTPSRoute() throws {
    let policy = NavigationPolicy(origin: URL(string: "https://platform.agpt.co")!)
    let attempt = AuthAttempt()
    let code = String(repeating: "a", count: 43)
    let query = "?code=\(code)&state=\(attempt.state)"
    let valid = URL(string: "https://platform.agpt.co/auth/mobile/callback" + query)!
    XCTAssertEqual(
      attempt.exchangeBody(callback: valid, policy: policy),
      "code=\(code)&verifier=\(attempt.verifier)")
    for value in [
      "https://platform.agpt.co/auth/mobile/callback?code=\(code)&state=wrong",
      valid.absoluteString + "&code=\(code)",
      "autogpt://auth/callback" + query,
      "https://platform.agpt.co.evil.com/auth/mobile/callback" + query,
      "https://platform.agpt.co:444/auth/mobile/callback" + query,
      "http://platform.agpt.co/auth/mobile/callback" + query,
      "https://platform.agpt.co/auth/mobile/callback/x" + query,
      valid.absoluteString + "#fragment",
    ] {
      XCTAssertNil(attempt.exchangeBody(callback: URL(string: value)!, policy: policy), value)
    }
  }

  func testAnExpiredAttemptCannotRedeemItsCallback() {
    let policy = NavigationPolicy(origin: URL(string: "https://platform.agpt.co")!)
    let attempt = AuthAttempt(createdAt: Date(timeIntervalSinceNow: -600))
    let code = String(repeating: "a", count: 43)
    let callback = URL(
      string:
        "https://platform.agpt.co/auth/mobile/callback?code=\(code)&state=\(attempt.state)")!
    XCTAssertNil(attempt.exchangeBody(callback: callback, policy: policy))
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
