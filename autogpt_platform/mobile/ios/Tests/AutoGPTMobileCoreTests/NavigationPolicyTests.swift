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
  let credentials = "https://user:password@platform.agpt.co"  // pragma: allowlist secret
  for value in [
    "http://platform.agpt.co", credentials,
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
  let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"  // pragma: allowlist secret
  let expected = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"  // pragma: allowlist secret
  #expect(PendingAuthentication.challenge(for: verifier) == expected)
}

@Test func checksStateBeforeAcceptingBrowserCancellation() throws {
  let pending = try PendingAuthentication()
  let cancelled = URL(string: "autogpt://auth/callback?error=access_denied&state=\(pending.state)")!
  #expect(throws: MobileError.authenticationCancelled) { try pending.validateCallback(cancelled) }
  let forged = URL(string: "autogpt://auth/callback?error=access_denied&state=wrong")!
  #expect(throws: MobileError.invalidCallback) { try pending.validateCallback(forged) }
}

@Test func acceptsOnlyUsableSameOriginHttpOnlyCookies() throws {
  let origin = try AppOrigin("https://platform.agpt.co")
  let header = "__Secure-better-auth.session_token=fixture; Path=/; Secure; HttpOnly; SameSite=Lax"
  let valid = HTTPCookie.cookies(withResponseHeaderFields: ["Set-Cookie": header], for: origin.url)
  #expect(valid.count == 1)
  #expect(origin.acceptsSessionCookie(valid[0]))
  for invalid in [
    header.replacingOccurrences(of: "; Secure", with: ""),
    header.replacingOccurrences(of: "; HttpOnly", with: ""),
    header.replacingOccurrences(of: "Path=/;", with: "Path=/api;"),
    header + "; Domain=agpt.co",
    header + "; Max-Age=0",
  ] {
    let cookies = HTTPCookie.cookies(
      withResponseHeaderFields: ["Set-Cookie": invalid], for: origin.url)
    #expect(cookies.allSatisfy { !origin.acceptsSessionCookie($0) })
  }
}

@Test func browserStartNeverContainsTheProofSecret() throws {
  let pending = try PendingAuthentication()
  let origin = try AppOrigin("https://platform.agpt.co")
  let url = pending.startURL(origin: origin)
  #expect(origin.contains(url))
  #expect(url.path == "/api/auth/mobile/start")
  #expect(!url.absoluteString.contains(pending.verifier))
  let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
  #expect(items?.count == 2)
  #expect(items?.first(where: { $0.name == "code_challenge" })?.value == pending.challenge)
}

@Test func rejectsAmbiguousSessionTokenCookies() throws {
  let origin = try AppOrigin("https://platform.agpt.co")
  let suffix = "; Path=/; Secure; HttpOnly"
  for names in [
    ["better-auth.session_token", "better-auth.session_token"],
    ["__Secure-better-auth.session_token", "__Secure-better-auth.session_token"],
    ["better-auth.session_token", "__Secure-better-auth.session_token"],
  ] {
    let header = "\(names[0])=first\(suffix), \(names[1])=second\(suffix)"
    let cookies = HTTPCookie.cookies(
      withResponseHeaderFields: ["Set-Cookie": header], for: origin.url)
    #expect(cookies.count == 2)
    #expect(origin.validatedSessionCookies(cookies) == nil)
  }
}

@Test func preservesCacheCookiesOnlyWithOneUsableSessionToken() throws {
  let origin = try AppOrigin("https://platform.agpt.co")
  let token = "__Secure-better-auth.session_token=fixture; Path=/; Secure; HttpOnly"
  let cache = "__Secure-better-auth.session_data=cache; Path=/; Secure; HttpOnly"
  let cookies = HTTPCookie.cookies(
    withResponseHeaderFields: ["Set-Cookie": "\(token), \(cache)"], for: origin.url)
  let accepted = try #require(origin.validatedSessionCookies(cookies))
  #expect(accepted.map(\.name) == cookies.map(\.name))
  for invalid in [cache, "\(token); Max-Age=0, \(cache)", "\(token); Max-Age=0, \(token)"] {
    let parsed = HTTPCookie.cookies(
      withResponseHeaderFields: ["Set-Cookie": invalid], for: origin.url)
    #expect(origin.validatedSessionCookies(parsed) == nil)
  }
}
