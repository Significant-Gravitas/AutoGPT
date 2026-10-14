import AVKit
import PDFKit
import QuickLook
import SwiftUI
import WebKit

/// Artifact content after an authorized fetch. Only the content reaches a
/// viewer; no viewer receives an app credential.
enum ArtifactContent {
  case html(String, allowsScripts: Bool)
  case text(String)
  case csv(String)
  case pdf(Data)
  case image(UIImage)
  case video(URL)
  /// Anything else, previewed or shared through the system.
  case file(URL)
}

struct ArtifactViewerView: View {
  let title: String
  let content: ArtifactContent
  @State private var externalLink: URL?
  @State private var previewing: URL?
  @Environment(\.openURL) private var openURL
  @Environment(\.dynamicTypeSize) private var typeSize

  var body: some View {
    viewer
      .navigationTitle(title)
      .navigationBarTitleDisplayMode(.inline)
      .confirmationDialog(
        "Open this link outside the artifact?",
        isPresented: Binding(get: { externalLink != nil }, set: { if !$0 { externalLink = nil } }),
        titleVisibility: .visible,
        presenting: externalLink
      ) { url in
        Button("Open in Safari") { openURL(url) }
        Button("Cancel", role: .cancel) {}
      } message: { url in
        Text(url.host ?? url.absoluteString)
      }
      .quickLookPreview($previewing)
  }

  @ViewBuilder
  private var viewer: some View {
    switch content {
    case .html(let html, let allowsScripts):
      if html.utf8.count > IsolatedArtifactWebView.maxBytes {
        ContentUnavailableView(
          "Too large to preview", systemImage: "doc.richtext",
          description: Text("This artifact is too large to open safely in the app."))
      } else {
        IsolatedArtifactWebView(html: html, allowsScripts: allowsScripts) { externalLink = $0 }
          .ignoresSafeArea(edges: .bottom)
          .accessibilityIdentifier("artifact-web-view")
      }
    case .text(let text):
      ScrollView {
        SelectableText(
          text: NSAttributedString(
            string: text,
            attributes: [
              .font: TextStyleFont.font(.callout, typeSize, monospaced: true),
              .foregroundColor: UIColor.label,
            ])
        )
        .padding()
      }
    case .csv(let text):
      CSVTableView(table: CSVTable(text))
    case .pdf(let data):
      PDFKitView(data: data).ignoresSafeArea(edges: .bottom)
    case .image(let image):
      ZoomableImage(image: image)
    case .video(let url):
      VideoPlayer(player: AVPlayer(url: url))
    case .file(let url):
      ContentUnavailableView {
        Label(url.lastPathComponent, systemImage: "doc")
      } description: {
        Text("This file type opens in the system previewer.")
      } actions: {
        Button("Preview") { previewing = url }
        ShareLink(item: url)
      }
    }
  }
}

/// HTML and React artifacts run here, isolated from the app: a fresh
/// nonpersistent website data store (no app cookies or storage), no script
/// message handlers, no file access (`about:blank` origin), a content security
/// policy and content blocker that stop every network request, no pop-ups or
/// permission grants, and no navigation away from the document. A tapped
/// http(s) link is handed back for an explicit confirmation instead.
struct IsolatedArtifactWebView: UIViewRepresentable {
  static let maxBytes = 2 * 1024 * 1024
  static let contentPolicy =
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:; connect-src 'none'; frame-src 'none'; child-src 'none'; worker-src 'none'; form-action 'none'; base-uri 'none'"
  private static let blockNetwork = """
    [{"trigger":{"url-filter":"^(https?|wss?|ftp)://"},"action":{"type":"block"}}]
    """

  let html: String
  let allowsScripts: Bool
  let onExternalLink: (URL) -> Void

  func makeCoordinator() -> Coordinator {
    Coordinator(onExternalLink: onExternalLink)
  }

  func makeUIView(context: Context) -> WKWebView {
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.defaultWebpagePreferences.allowsContentJavaScript = allowsScripts
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.dataDetectorTypes = []
    configuration.mediaTypesRequiringUserActionForPlayback = .all
    configuration.allowsInlineMediaPlayback = false
    let view = WKWebView(frame: .zero, configuration: configuration)
    view.navigationDelegate = context.coordinator
    view.uiDelegate = context.coordinator
    view.allowsLinkPreview = false
    view.allowsBackForwardNavigationGestures = false
    view.isInspectable = false
    let document = Self.document(html)
    WKContentRuleListStore.default().compileContentRuleList(
      forIdentifier: "autogpt.artifact.block-network", encodedContentRuleList: Self.blockNetwork
    ) { [weak view] list, _ in
      // WebKit completes on the main thread.
      MainActor.assumeIsolated {
        guard let view else { return }
        // Without the blocker the CSP still applies; the document loads either way.
        if let list { view.configuration.userContentController.add(list) }
        view.loadHTMLString(document, baseURL: nil)
      }
    }
    return view
  }

  func updateUIView(_ view: WKWebView, context: Context) {}

  static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) {
    view.stopLoading()
    view.navigationDelegate = nil
    view.uiDelegate = nil
  }

  /// Puts the content policy first in the document head, before any artifact
  /// markup can run.
  static func document(_ html: String) -> String {
    let meta = "<meta http-equiv=\"Content-Security-Policy\" content=\"\(contentPolicy)\">"
    if let head = html.range(of: "<head[^>]*>", options: [.regularExpression, .caseInsensitive]) {
      var document = html
      document.insert(contentsOf: meta, at: head.upperBound)
      return document
    }
    if let root = html.range(of: "<html[^>]*>", options: [.regularExpression, .caseInsensitive]) {
      var document = html
      document.insert(contentsOf: "<head>\(meta)</head>", at: root.upperBound)
      return document
    }
    return "<!doctype html><html><head>\(meta)</head><body>\(html)</body></html>"
  }

  final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
    private let onExternalLink: (URL) -> Void
    private var loadedDocument = false

    init(onExternalLink: @escaping (URL) -> Void) {
      self.onExternalLink = onExternalLink
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction) async
      -> WKNavigationActionPolicy
    {
      let url = navigationAction.request.url
      let isMainFrame = navigationAction.targetFrame?.isMainFrame ?? false
      if !loadedDocument, isMainFrame, url?.absoluteString == "about:blank" {
        loadedDocument = true
        return .allow
      }
      if navigationAction.navigationType == .linkActivated, let url,
        let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http"
      {
        onExternalLink(url)
      }
      return .cancel
    }

    func webView(
      _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
      for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
      nil
    }

    func webView(
      _ webView: WKWebView, decideMediaCapturePermissionsFor origin: WKSecurityOrigin,
      initiatedBy frame: WKFrameInfo, type: WKMediaCaptureType
    ) async -> WKPermissionDecision {
      .deny
    }

    func webView(
      _ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
      initiatedByFrame frame: WKFrameInfo
    ) async {}

    func webView(
      _ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
      initiatedByFrame frame: WKFrameInfo
    ) async -> Bool {
      false
    }

    func webView(
      _ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
      defaultText: String?, initiatedByFrame frame: WKFrameInfo
    ) async -> String? {
      nil
    }
  }
}

struct PDFKitView: UIViewRepresentable {
  let data: Data

  func makeUIView(context: Context) -> PDFView {
    let view = PDFView()
    view.autoScales = true
    view.displayMode = .singlePageContinuous
    view.document = PDFDocument(data: data)
    return view
  }

  func updateUIView(_ view: PDFView, context: Context) {}
}

/// A bounded CSV table. Cells are shown as text; nothing is evaluated.
struct CSVTable: Equatable {
  static let maxRows = 200
  static let maxColumns = 30
  static let maxCell = 500

  let header: [String]
  let rows: [[String]]
  let truncated: Bool

  init(_ text: String) {
    let parsed = Self.parse(text)
    let width = min(parsed.map(\.count).max() ?? 0, Self.maxColumns)
    let normalize = { (row: [String]) -> [String] in
      let cells = row.prefix(width).map { $0.truncated(to: Self.maxCell) }
      return cells + Array(repeating: "", count: max(0, width - cells.count))
    }
    header = parsed.first.map(normalize) ?? []
    let body = parsed.dropFirst()
    rows = body.prefix(Self.maxRows).map(normalize)
    truncated = body.count > Self.maxRows || (parsed.map(\.count).max() ?? 0) > Self.maxColumns
  }

  private static func parse(_ text: String) -> [[String]] {
    var rows: [[String]] = []
    var row: [String] = []
    var field = ""
    var quoted = false
    var iterator = Array(text).makeIterator()
    var pending: Character? = nil
    func next() -> Character? {
      if let character = pending {
        pending = nil
        return character
      }
      return iterator.next()
    }
    while let character = next() {
      if quoted {
        if character == "\"" {
          if let following = next() {
            if following == "\"" {
              field.append("\"")
            } else {
              quoted = false
              pending = following
            }
          } else {
            quoted = false
          }
        } else {
          field.append(character)
        }
        continue
      }
      switch character {
      case "\"": quoted = true
      case ",":
        row.append(field)
        field = ""
      case "\n", "\r\n", "\r":
        row.append(field)
        field = ""
        rows.append(row)
        row = []
        if rows.count > maxRows { return rows }
      default: field.append(character)
      }
    }
    if !field.isEmpty || !row.isEmpty {
      row.append(field)
      rows.append(row)
    }
    return rows
  }
}

struct CSVTableView: View {
  let table: CSVTable

  var body: some View {
    ScrollView([.horizontal, .vertical]) {
      Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
        GridRow {
          ForEach(Array(table.header.enumerated()), id: \.offset) { _, title in
            Text(title).font(.subheadline.weight(.semibold)).accessibilityAddTraits(.isHeader)
          }
        }
        Divider()
        ForEach(Array(table.rows.enumerated()), id: \.offset) { _, row in
          GridRow {
            ForEach(Array(row.enumerated()), id: \.offset) { column, value in
              Text(value)
                .font(.subheadline)
                .frame(maxWidth: 240, alignment: .leading)
                .accessibilityLabel(
                  column < table.header.count ? "\(table.header[column]): \(value)" : value)
            }
          }
        }
        if table.truncated {
          Text("Showing the first \(CSVTable.maxRows) rows and \(CSVTable.maxColumns) columns.")
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
      }
      .padding()
      .textSelection(.enabled)
    }
  }
}

private struct ZoomableImage: View {
  let image: UIImage
  @State private var scale: CGFloat = 1
  @GestureState private var pinch: CGFloat = 1

  var body: some View {
    ScrollView([.horizontal, .vertical]) {
      Image(uiImage: image)
        .resizable()
        .scaledToFit()
        .scaleEffect(scale * pinch)
        .gesture(
          MagnifyGesture()
            .updating($pinch) { value, state, _ in state = value.magnification }
            .onEnded { value in scale = min(max(scale * value.magnification, 1), 6) }
        )
        .accessibilityLabel("Image")
    }
  }
}
