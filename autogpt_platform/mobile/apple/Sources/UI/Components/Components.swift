import ImageIO
import SwiftUI
import UIKit

enum Brand {
  static let prominent = Color.accentColor
  static let onProminent = Color(uiColor: .systemBackground)
  static let surface = Color(
    uiColor: UIColor { traits in
      traits.userInterfaceStyle == .dark
        ? UIColor(red: 0.153, green: 0.153, blue: 0.165, alpha: 1)
        : UIColor(red: 0.957, green: 0.957, blue: 0.961, alpha: 1)
    })
  static let border = Color(
    uiColor: UIColor { traits in
      traits.userInterfaceStyle == .dark
        ? UIColor(red: 0.247, green: 0.247, blue: 0.275, alpha: 1)
        : UIColor(red: 0.894, green: 0.894, blue: 0.906, alpha: 1)
    })
  static let portraitBackground = Color(
    uiColor: UIColor { traits in
      traits.userInterfaceStyle == .dark
        ? UIColor(red: 0.18, green: 0.15, blue: 0.24, alpha: 1)
        : UIColor(red: 0.96, green: 0.95, blue: 1, alpha: 1)
    })
}

/// A participant's portrait, falling back to their initial when there is no usable image.
struct AvatarView: View {
  let name: String
  let url: URL?
  var size: CGFloat = 44
  var asset: String? = nil
  @State private var portrait: UIImage?

  var body: some View {
    Group {
      if let asset {
        Image(asset).resizable().scaledToFill()
      } else if let portrait {
        Image(uiImage: portrait).resizable().scaledToFill()
      } else {
        initial
      }
    }
    .frame(width: size, height: size)
    .clipShape(Circle())
    .accessibilityHidden(true)
    .task(id: url) {
      portrait = nil
      guard asset == nil, let url else { return }
      do {
        let data = try await HTTPTransport.shared.publicImage(url)
        try Task.checkCancellation()
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0,
            [
              kCGImageSourceCreateThumbnailFromImageAlways: true,
              kCGImageSourceThumbnailMaxPixelSize: 512,
              kCGImageSourceCreateThumbnailWithTransform: true,
            ] as CFDictionary)
        else { return }
        portrait = UIImage(cgImage: image)
      } catch {}
    }
  }

  private var initial: some View {
    ZStack {
      Circle().fill(Color(uiColor: .tertiarySystemFill))
      Text(name.trimmed.first.map { String($0).uppercased() } ?? "?")
        .font(.system(size: size * 0.42, weight: .semibold))
        .foregroundStyle(.secondary)
        .dynamicTypeSize(.large)
    }
  }
}

struct OttoArtwork: View {
  var size: CGFloat = 140

  var body: some View {
    Image("OttoWelcome")
      .resizable()
      .scaledToFit()
      .frame(width: size, height: size)
      .background(Brand.portraitBackground, in: Circle())
      .accessibilityHidden(true)
  }
}

/// A short status line under a list, in the style of grouped footers.
struct InlineNotice: View {
  let text: String
  var tone: Tone = .neutral

  enum Tone { case neutral, danger }

  var body: some View {
    Label(text, systemImage: tone == .danger ? "exclamationmark.circle" : "info.circle")
      .font(.footnote)
      .foregroundStyle(tone == .danger ? Color.red : Color.secondary)
      .frame(maxWidth: .infinity, alignment: .leading)
  }
}

struct FieldRow<Field: View>: View {
  let label: String
  @ViewBuilder let field: Field

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(label)
        .font(.footnote)
        .foregroundStyle(.secondary)
        .accessibilityHidden(true)
      field
    }
    .padding(.vertical, 2)
  }
}

/// Text with native, partial selection. SwiftUI `Text` on iOS only offers
/// copying the whole string, so message text is a non-editable UITextView.
/// Each Markdown block is its own view, which also keeps VoiceOver
/// navigation paragraph by paragraph.
struct SelectableText: UIViewRepresentable {
  let text: NSAttributedString
  var hugsContent = false
  var isHeader = false

  func makeUIView(context: Context) -> UITextView {
    let view = UITextView()
    view.isEditable = false
    view.isSelectable = true
    view.isScrollEnabled = false
    view.backgroundColor = .clear
    view.textContainerInset = .zero
    view.textContainer.lineFragmentPadding = 0
    view.adjustsFontForContentSizeCategory = true
    view.dataDetectorTypes = []
    view.linkTextAttributes = [
      .foregroundColor: UIColor.link, .underlineStyle: NSUnderlineStyle.single.rawValue,
    ]
    view.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    view.setContentHuggingPriority(.defaultHigh, for: .vertical)
    return view
  }

  func updateUIView(_ view: UITextView, context: Context) {
    if !view.attributedText.isEqual(to: text) { view.attributedText = text }
    if isHeader {
      view.accessibilityTraits.insert(.header)
    } else {
      view.accessibilityTraits.remove(.header)
    }
  }

  func sizeThatFits(_ proposal: ProposedViewSize, uiView: UITextView, context: Context) -> CGSize? {
    let proposed = proposal.width.flatMap { $0.isFinite ? $0 : nil }
    var width = proposed ?? 320
    if hugsContent {
      // Unbounded (a horizontal scroll view): code keeps its own line breaks.
      let limit = proposed ?? .greatestFiniteMagnitude
      let natural = text.boundingRect(
        with: CGSize(width: limit, height: .greatestFiniteMagnitude),
        options: [.usesLineFragmentOrigin, .usesFontLeading], context: nil
      ).width
      width = min(limit, ceil(natural) + 2)
    }
    let fitted = uiView.sizeThatFits(CGSize(width: width, height: .greatestFiniteMagnitude))
    return CGSize(width: width, height: ceil(fitted.height))
  }
}

enum TextStyleFont {
  static func font(
    _ style: UIFont.TextStyle, _ size: DynamicTypeSize, bold: Bool = false, monospaced: Bool = false
  )
    -> UIFont
  {
    let traits = UITraitCollection(preferredContentSizeCategory: category(size))
    let base = UIFont.preferredFont(forTextStyle: style, compatibleWith: traits)
    if monospaced {
      let mono = UIFont.monospacedSystemFont(
        ofSize: base.pointSize * 0.9, weight: bold ? .semibold : .regular)
      return mono
    }
    guard bold, let descriptor = base.fontDescriptor.withSymbolicTraits(.traitBold) else {
      return base
    }
    return UIFont(descriptor: descriptor, size: base.pointSize)
  }

  static func category(_ size: DynamicTypeSize) -> UIContentSizeCategory {
    switch size {
    case .xSmall: .extraSmall
    case .small: .small
    case .medium: .medium
    case .large: .large
    case .xLarge: .extraLarge
    case .xxLarge: .extraExtraLarge
    case .xxxLarge: .extraExtraExtraLarge
    case .accessibility1: .accessibilityMedium
    case .accessibility2: .accessibilityLarge
    case .accessibility3: .accessibilityExtraLarge
    case .accessibility4: .accessibilityExtraExtraLarge
    case .accessibility5: .accessibilityExtraExtraExtraLarge
    @unknown default: .large
    }
  }
}

/// Native rendering of the Markdown subset: every block is selectable text,
/// links open only for http(s), and code scrolls horizontally.
struct MarkdownView: View {
  let text: String
  var secondary = false
  @Environment(\.dynamicTypeSize) private var typeSize

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      ForEach(Array(Markdown.parse(text).enumerated()), id: \.offset) { _, block in
        blockView(block)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  @ViewBuilder
  private func blockView(_ block: MarkdownBlock) -> some View {
    switch block {
    case .paragraph(let inline):
      SelectableText(text: attributed(inline, style: .body))
    case .heading(let level, let inline):
      SelectableText(
        text: attributed(inline, style: level <= 2 ? .title3 : .headline, bold: true),
        isHeader: true)
    case .code(_, let code), .preformatted(let code):
      ScrollView(.horizontal) {
        SelectableText(
          text: plain(code, font: TextStyleFont.font(.callout, typeSize, monospaced: true)),
          hugsContent: true
        )
        .padding(12)
      }
      .background(Color(uiColor: .tertiarySystemFill), in: RoundedRectangle(cornerRadius: 8))
    case .list(let ordered, let start, let items):
      VStack(alignment: .leading, spacing: 4) {
        ForEach(Array(items.enumerated()), id: \.offset) { index, item in
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(ordered ? "\(start + index)." : "•")
              .font(.body)
              .foregroundStyle(secondary ? .secondary : .primary)
              .frame(minWidth: 18, alignment: .trailing)
              .accessibilityHidden(true)
            SelectableText(text: attributed(item, style: .body))
          }
        }
      }
    case .quote(let inline):
      HStack(spacing: 10) {
        RoundedRectangle(cornerRadius: 1.5).fill(Color(uiColor: .separator)).frame(width: 3)
        SelectableText(text: attributed(inline, style: .body, color: .secondaryLabel))
      }
    case .rule:
      Divider().padding(.vertical, 4)
    }
  }

  private func plain(_ text: String, font: UIFont) -> NSAttributedString {
    NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: UIColor.label])
  }

  private func attributed(
    _ nodes: [Inline], style: UIFont.TextStyle, bold: Bool = false, color: UIColor? = nil
  ) -> NSAttributedString {
    let base = TextStyleFont.font(style, typeSize, bold: bold)
    let result = NSMutableAttributedString()
    append(
      nodes, to: result, font: base, color: color ?? (secondary ? .secondaryLabel : .label),
      bold: bold, italic: false, strike: false, link: nil)
    return result
  }

  private func append(
    _ nodes: [Inline], to result: NSMutableAttributedString, font: UIFont, color: UIColor,
    bold: Bool,
    italic: Bool, strike: Bool, link: URL?
  ) {
    for node in nodes {
      switch node {
      case .text(let text):
        result.append(
          run(
            text, font: font, color: color, bold: bold, italic: italic, strike: strike, link: link))
      case .code(let text):
        let mono = UIFont.monospacedSystemFont(ofSize: font.pointSize * 0.9, weight: .regular)
        var codeAttributes = attributes(font: mono, color: color, strike: strike, link: link)
        codeAttributes[.backgroundColor] = UIColor.tertiarySystemFill
        result.append(NSAttributedString(string: text, attributes: codeAttributes))
      case .strong(let children):
        append(
          children, to: result, font: font, color: color, bold: true, italic: italic,
          strike: strike, link: link)
      case .emphasis(let children):
        append(
          children, to: result, font: font, color: color, bold: bold, italic: true, strike: strike,
          link: link)
      case .strike(let children):
        append(
          children, to: result, font: font, color: color, bold: bold, italic: italic, strike: true,
          link: link)
      case .link(let url, let children):
        append(
          children, to: result, font: font, color: color, bold: bold, italic: italic,
          strike: strike, link: url ?? link)
      }
    }
  }

  private func run(
    _ text: String, font: UIFont, color: UIColor, bold: Bool, italic: Bool, strike: Bool, link: URL?
  ) -> NSAttributedString {
    var traits: UIFontDescriptor.SymbolicTraits = []
    if bold { traits.insert(.traitBold) }
    if italic { traits.insert(.traitItalic) }
    var styled = font
    if !traits.isEmpty,
      let descriptor = font.fontDescriptor.withSymbolicTraits(
        font.fontDescriptor.symbolicTraits.union(traits))
    {
      styled = UIFont(descriptor: descriptor, size: font.pointSize)
    }
    return NSAttributedString(
      string: text, attributes: attributes(font: styled, color: color, strike: strike, link: link))
  }

  private func attributes(font: UIFont, color: UIColor, strike: Bool, link: URL?)
    -> [NSAttributedString.Key: Any]
  {
    var attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: color]
    if strike { attributes[.strikethroughStyle] = NSUnderlineStyle.single.rawValue }
    if let link { attributes[.link] = link }
    return attributes
  }
}
