import Foundation
import Testing

@testable import AutoGPTMobileCore

@Test func acceptsOnlyConfiguredOrigin() throws {
  let origin = try AppOrigin("https://platform.agpt.co")
  #expect(origin.contains(URL(string: "https://platform.agpt.co/copilot?session=abc")!))
  #expect(origin.contains(URL(string: "https://platform.agpt.co:443/login")!))
  for value in [
    "http://platform.agpt.co/copilot", "https://platform.agpt.co.evil.example/",
    "https://evil.example/?next=platform.agpt.co", "https://platform.agpt.co:444/",
    "https://user@platform.agpt.co/", "file:///etc/passwd", "javascript:alert(1)",
  ] {
    #expect(!origin.contains(URL(string: value)!))
  }
}

@Test func validatesConfiguredAddress() throws {
  for value in [
    "http://platform.agpt.co", "https://user:password@platform.agpt.co",  // pragma: allowlist secret
    "https://platform.agpt.co/copilot", "https://platform.agpt.co?evil=1",
    "https://platform.agpt.co#fragment", "file:///tmp/app", "not a URL",
  ] {
    #expect(throws: Error.self) { try AppOrigin(value) }
  }
  #expect(
    try AppOrigin("https://PLATFORM.agpt.co/").url.absoluteString == "https://platform.agpt.co")
  #expect(throws: Error.self) { try AppOrigin("http://localhost:8765") }
  #expect(try AppOrigin("http://localhost:8765", allowLocalHTTP: true).url.port == 8765)
  #expect(throws: Error.self) {
    try AppOrigin("http://localhost.evil.example", allowLocalHTTP: true)
  }
}

@Test func confinesBlobDownloadsToTheCurrentOrigin() throws {
  let origin = try AppOrigin("https://platform.agpt.co")
  #expect(origin.allowsBlobDownload(URL(string: "blob:https://platform.agpt.co/1234")!))
  #expect(!origin.allowsBlobDownload(URL(string: "blob:https://evil.example/1234")!))
  #expect(!origin.allowsBlobDownload(URL(string: "data:text/html,evil")!))
}

@Test func validatesAuthenticationProofAndCallback() throws {
  let pending = try PendingAuthentication(now: Date(timeIntervalSince1970: 100))
  #expect(pending.verifier.count == 43)
  #expect(pending.challenge.count == 43)
  #expect(pending.state.count == 43)
  #expect(pending.verifier != pending.state)
  let code = String(repeating: "A", count: 43)
  let valid = URL(string: "autogpt://auth/callback?code=\(code)&state=\(pending.state)")!
  #expect(try pending.validateCallback(valid, now: Date(timeIntervalSince1970: 101)) == code)
  for value in [
    "autogpt://evil/callback?code=\(code)&state=\(pending.state)",
    "autogpt://auth/callback?code=\(code)&state=wrong",
    "autogpt://auth/callback?code=\(code)&state=\(pending.state)&state=other",
    "autogpt://auth/callback?code=\(code)&state=\(pending.state)#extra",
    "https://auth/callback?code=\(code)&state=\(pending.state)",
  ] {
    #expect(throws: Error.self) {
      try pending.validateCallback(URL(string: value)!, now: Date(timeIntervalSince1970: 101))
    }
  }
  #expect(throws: Error.self) {
    try pending.validateCallback(valid, now: Date(timeIntervalSince1970: 701))
  }
}

@Test func usesRFC7636Challenge() {
  #expect(
    PendingAuthentication.challenge(for: "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")  // pragma: allowlist secret
      == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")  // pragma: allowlist secret
}
