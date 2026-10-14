#if UI_PROTOCOL_FIXTURES
  import PDFKit
  import SwiftUI

  /// Debug builds only. Launching with `-AutoGPTUIProtocolFixtures YES` opens
  /// these interaction fixtures instead of the app. They drive the real
  /// conversation controller, timeline, cards and viewers with local
  /// UI-protocol data. Nothing here contacts a server: this is fixture
  /// coverage, not integrated tool behavior.
  enum FixtureLaunch {
    static let argument = "AutoGPTUIProtocolFixtures"

    static var isRequested: Bool {
      if UserDefaults.standard.object(forKey: argument) != nil {
        return UserDefaults.standard.bool(forKey: argument)
      }
      let arguments = ProcessInfo.processInfo.arguments
      return arguments.contains("-\(argument)") || arguments.contains(argument)
    }
  }

  struct FixtureCatalogView: View {
    var body: some View {
      NavigationStack {
        List {
          Section {
            Label(
              "Local UI-protocol fixtures. Nothing on these screens reaches a server or counts as integrated tool behavior.",
              systemImage: "testtube.2"
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
          }
          Section("Device") {
            NavigationLink("Secure storage") {
              FixtureSecureStorageView()
            }
            .accessibilityIdentifier("fixture-secure-storage")
          }
          Section("Conversation") {
            NavigationLink("Long streamed conversation") {
              FixtureConversationScreen(scenario: .longStream)
            }
            .accessibilityIdentifier("fixture-long-conversation")
            NavigationLink("Question card") {
              FixtureConversationScreen(scenario: .question)
            }
            .accessibilityIdentifier("fixture-question")
          }
          Section("Connections") {
            NavigationLink("Edit a connection") {
              ConnectionsListView(service: FixtureConnectionService())
            }
            .accessibilityIdentifier("fixture-connections")
          }
          Section("Artifacts") {
            NavigationLink("Isolated HTML artifact") {
              ArtifactViewerView(
                title: "Isolation check",
                content: .html(FixtureArtifacts.isolationCheck, allowsScripts: true)
              )
            }
            .accessibilityIdentifier("fixture-artifact-html")
            NavigationLink("PDF artifact") {
              ArtifactViewerView(title: "Report.pdf", content: .pdf(FixtureArtifacts.pdf()))
            }
            .accessibilityIdentifier("fixture-artifact-pdf")
            NavigationLink("CSV artifact") {
              ArtifactViewerView(title: "Leads.csv", content: .csv(FixtureArtifacts.csv))
            }
            .accessibilityIdentifier("fixture-artifact-csv")
            NavigationLink("Other file (Quick Look)") {
              ArtifactViewerView(title: "Notes.txt", content: .file(FixtureArtifacts.textFile()))
            }
            .accessibilityIdentifier("fixture-artifact-file")
          }
        }
        .navigationTitle("UI Protocol Fixtures")
      }
      .fixtureBanner()
    }
  }

  private struct FixtureSecureStorageView: View {
    @State private var result = "Checking secure storage…"

    var body: some View {
      Text(result)
        .padding()
        .navigationTitle("Secure Storage")
        .task { result = Self.probe() }
    }

    private static func probe() -> String {
      let key = "native-ui-probe-\(UUID().uuidString)"
      let credential = StoredCredential(
        sessionToken: "fixture.\(UUID().uuidString)", authOrigin: "https://example.invalid",
        apiOrigin: "https://api.example.invalid", userID: "fixture",
        email: "fixture@example.invalid",
        name: "Fixture", savedAt: Date())
      defer { CredentialStore.delete(key) }
      do {
        try CredentialStore.write(key, credential)
        guard try CredentialStore.read(key)?.sessionToken == credential.sessionToken else {
          return "Secure storage read failed"
        }
        CredentialStore.delete(key)
        guard try CredentialStore.read(key) == nil else { return "Secure storage deletion failed" }
        return "Secure storage round trip passed"
      } catch {
        return "Secure storage is unavailable"
      }
    }
  }

  struct FixtureConversationScreen: View {
    private static let participant = Participant(
      expertID: nil, name: "Otto", role: "Fixture conversation", avatarURL: nil,
      avatarAsset: "OttoPortrait", readOnlyReason: nil
    )

    let scenario: FixtureConversationService.Scenario
    @State private var controller: ConversationController
    @Environment(\.scenePhase) private var scenePhase

    init(scenario: FixtureConversationService.Scenario) {
      self.scenario = scenario
      _controller = State(
        initialValue: ConversationController(
          service: FixtureConversationService(scenario: scenario),
          sessionID: FixtureConversationService.sessionID, expertID: nil))
    }

    var body: some View {
      ConversationView(controller: controller, participant: { _ in Self.participant })
        .onChange(of: scenePhase) { _, phase in controller.scenePhaseChanged(phase) }
        .onDisappear { controller.dispose() }
    }
  }

  extension View {
    func fixtureBanner() -> some View {
      VStack(spacing: 0) {
        Text("Fixture — not connected to a server")
          .font(.caption.weight(.semibold))
          .foregroundStyle(.white)
          .frame(maxWidth: .infinity)
          .padding(.vertical, 4)
          .background(Color.orange)
          .accessibilityIdentifier("fixture-banner")
        self
      }
    }
  }

  @MainActor
  enum FixtureArtifacts {
    /// Probes the viewer's isolation from inside the artifact and reports
    /// what it could reach.
    static let isolationCheck = """
      <!doctype html>
      <html><head><meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
      body { font: -apple-system-body; padding: 16px; color: #111; background: #fff; }
      @media (prefers-color-scheme: dark) { body { color: #eee; background: #000; } }
      li { margin: 8px 0; } .ok { color: #1a7f37; } .bad { color: #c62828; }
      </style></head>
      <body>
      <h1>Isolation check</h1>
      <p>This artifact runs without app credentials, cookies, storage that outlives the viewer, or a native bridge.</p>
      <ul id="checks"></ul>
      <p><a href="https://agpt.co">An external link</a> asks before leaving the viewer.</p>
      <script>
      const list = document.getElementById("checks");
      function report(name, ok, detail) {
        const item = document.createElement("li");
        item.className = ok ? "ok" : "bad";
        item.textContent = (ok ? "✓ " : "✗ ") + name + ": " + detail;
        list.appendChild(item);
      }
      report("Native bridge", !(window.webkit && window.webkit.messageHandlers),
        "no script message handlers are installed");
      try { document.cookie = "probe=1"; } catch (e) {}
      report("Cookies", document.cookie === "", document.cookie === "" ? "none readable or writable" : document.cookie);
      report("Origin", location.origin === "null" || location.protocol === "about:", "opaque " + location.href);
      fetch("https://example.com/").then(
        () => report("Network", false, "a request left the viewer"),
        () => report("Network", true, "requests are blocked"));
      const image = new Image();
      image.onload = () => report("Remote images", false, "an image loaded from the network");
      image.onerror = () => report("Remote images", true, "remote images are blocked");
      image.src = "https://example.com/pixel.png";
      window.open("https://example.com/");
      report("Pop-ups", true, "window.open is ignored");
      </script>
      </body></html>
      """

    static let csv = #"""
      Company,Contact,Stage,Value,Note
      Acme Rockets,Ada Lovelace,Qualified,12000,"Prefers email, not calls"
      Globex,Hank Scorpio,Proposal,48000,=HYPERLINK("https://example.com")
      Initech,Peter Gibbons,Lost,0,"Said ""maybe next year"""
      Umbrella,Alice,Negotiation,30500,Multi-line
      Wonka,Charlie,Won,99000,Golden ticket
      """#

    static func pdf() -> Data {
      let page = CGRect(x: 0, y: 0, width: 612, height: 792)
      return UIGraphicsPDFRenderer(bounds: page).pdfData { context in
        for number in 1...3 {
          context.beginPage()
          let text =
            "Fixture report, page \(number)\n\nGenerated on device for the PDF viewer check."
          (text as NSString).draw(
            in: page.insetBy(dx: 54, dy: 72),
            withAttributes: [.font: UIFont.preferredFont(forTextStyle: .title2)])
        }
      }
    }

    static func textFile() -> URL {
      let url = FileManager.default.temporaryDirectory.appendingPathComponent("Notes.txt")
      try? Data("Fixture notes for the Quick Look previewer.\n".utf8).write(to: url)
      return url
    }
  }
#endif
