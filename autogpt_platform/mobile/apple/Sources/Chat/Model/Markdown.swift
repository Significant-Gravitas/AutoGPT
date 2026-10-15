import Foundation

/// A deliberately small Markdown subset for assistant replies, rendered with
/// native text. Anything outside the subset stays readable as plain text.
/// Links are only ever made tappable for http(s) targets.
indirect enum Inline: Equatable, Sendable {
  case text(String)
  case code(String)
  case strong([Inline])
  case emphasis([Inline])
  case strike([Inline])
  case link(URL?, [Inline])
}

enum MarkdownBlock: Equatable, Sendable {
  case paragraph([Inline])
  case heading(level: Int, [Inline])
  case code(language: String?, text: String)
  case list(ordered: Bool, start: Int, items: [[Inline]])
  case quote([Inline])
  case preformatted(String)
  case rule
}

enum Markdown {
  private static let fence = regex(#"^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$"#)
  private static let heading = regex(#"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$"#)
  private static let rule = regex(#"^\s{0,3}([-*_])(\s*\1){2,}\s*$"#)
  private static let listItem = regex(#"^(\s*)([-*+]|(\d{1,9})[.)])\s+(.*)$"#)
  private static let quote = regex(#"^\s{0,3}>\s?(.*)$"#)
  private static let tableDivider = regex(
    #"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$"#)
  private static let inline = regex(
    #"(`[^`\n]+`)|(\[[^\]\n]+\]\([^)\s]+(?:\s+"[^"]*")?\))|(\*\*[^*\n]+?\*\*)|(__[^_\n]+?__)|(~~[^~\n]+?~~)|(\*[^*\s][^*\n]*?\*)|((?<![\w])_[^_\s][^_\n]*?_(?![\w]))|(https?://[^\s<>()]+[^\s<>().,;:!?'"])"#
  )
  private static let linkParts = regex(#"^\[([^\]]+)\]\(([^)\s]+)"#)

  static func parse(_ source: String) -> [MarkdownBlock] {
    let lines = source.replacingOccurrences(of: "\r\n", with: "\n")
      .replacingOccurrences(of: "\r", with: "\n")
      .components(separatedBy: "\n")
    var blocks: [MarkdownBlock] = []
    var index = 0
    while index < lines.count {
      let line = lines[index]
      if line.trimmed.isEmpty {
        index += 1
        continue
      }
      if let groups = match(fence, line) {
        let marker = groups[0] ?? "```"
        var body: [String] = []
        index += 1
        // An unterminated fence (a reply still streaming) runs to the end.
        while index < lines.count, !lines[index].trimmed.hasPrefix(marker) {
          body.append(lines[index])
          index += 1
        }
        index += 1
        let language = groups[1].flatMap { $0.isEmpty ? nil : $0 }
        blocks.append(.code(language: language, text: body.joined(separator: "\n")))
        continue
      }
      if let groups = match(heading, line) {
        blocks.append(.heading(level: groups[0]?.count ?? 1, parseInline(groups[1] ?? "")))
        index += 1
        continue
      }
      if matches(rule, line) {
        blocks.append(.rule)
        index += 1
        continue
      }
      if isTableStart(lines, index) {
        var rows: [String] = []
        while index < lines.count, lines[index].contains("|") {
          rows.append(lines[index].trimmed)
          index += 1
        }
        blocks.append(.preformatted(rows.joined(separator: "\n")))
        continue
      }
      if matches(listItem, line) {
        index = parseList(lines, index, &blocks)
        continue
      }
      if matches(quote, line) {
        var quoted: [String] = []
        while index < lines.count, let groups = match(quote, lines[index]) {
          quoted.append(groups[0] ?? "")
          index += 1
        }
        blocks.append(.quote(parseInline(quoted.joined(separator: "\n"))))
        continue
      }
      // The first line is always consumed, so the loop can't stall.
      var paragraph = [line.trimmed]
      index += 1
      while index < lines.count, !lines[index].trimmed.isEmpty, !startsBlock(lines, index) {
        paragraph.append(lines[index].trimmed)
        index += 1
      }
      blocks.append(.paragraph(parseInline(paragraph.joined(separator: "\n"))))
    }
    return blocks
  }

  static func parseInline(_ text: String) -> [Inline] {
    var result: [Inline] = []
    let nsText = text as NSString
    var last = 0
    for match in inline.matches(in: text, range: NSRange(location: 0, length: nsText.length)) {
      if match.range.location > last {
        result.append(
          .text(
            nsText.substring(with: NSRange(location: last, length: match.range.location - last))))
      }
      result.append(token(match, nsText))
      last = match.range.location + match.range.length
    }
    if last < nsText.length { result.append(.text(nsText.substring(from: last))) }
    return result
  }

  /// The visible text of inline content, for accessibility and copying.
  static func plainText(_ nodes: [Inline]) -> String {
    nodes.map { node in
      switch node {
      case .text(let text), .code(let text): text
      case .strong(let children), .emphasis(let children), .strike(let children),
        .link(_, let children):
        plainText(children)
      }
    }.joined()
  }

  private static func token(_ match: NSTextCheckingResult, _ text: NSString) -> Inline {
    func group(_ index: Int) -> String? {
      let range = match.range(at: index)
      return range.location == NSNotFound ? nil : text.substring(with: range)
    }
    let whole = text.substring(with: match.range)
    if let code = group(1) { return .code(String(code.dropFirst().dropLast())) }
    if let link = group(2), let parts = Markdown.match(linkParts, link) {
      return .link(safeURL(parts[1] ?? ""), parseInline(parts[0] ?? ""))
    }
    if group(3) != nil || group(4) != nil {
      return .strong(parseInline(String(whole.dropFirst(2).dropLast(2))))
    }
    if group(5) != nil { return .strike(parseInline(String(whole.dropFirst(2).dropLast(2)))) }
    if group(6) != nil || group(7) != nil {
      return .emphasis(parseInline(String(whole.dropFirst().dropLast())))
    }
    if let url = group(8) { return .link(safeURL(url), [.text(url)]) }
    return .text(whole)
  }

  /// Only absolute http(s) links open; workspace, script and app links stay inert.
  static func safeURL(_ raw: String) -> URL? {
    guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(),
      scheme == "https" || scheme == "http", url.host?.isEmpty == false
    else { return nil }
    return url
  }

  private static func parseList(_ lines: [String], _ start: Int, _ blocks: inout [MarkdownBlock])
    -> Int
  {
    let first = match(listItem, lines[start]) ?? []
    let ordered = first.count > 2 && first[2] != nil
    var items: [String] = []
    var index = start
    while index < lines.count {
      let line = lines[index]
      let item = match(listItem, line)
      if let item, (item[2] != nil) == ordered {
        items.append(item[3] ?? "")
      } else if item != nil || line.trimmed.isEmpty || startsBlock(lines, index) {
        // A blank line followed by another item continues the same list.
        if line.trimmed.isEmpty, index + 1 < lines.count,
          let next = match(listItem, lines[index + 1]),
          (next[2] != nil) == ordered
        {
          index += 1
          continue
        }
        break
      } else if !items.isEmpty {
        items[items.count - 1] += "\n\(line.trimmed)"
      }
      index += 1
    }
    let startNumber = ordered ? Int(first[2] ?? "1") ?? 1 : 1
    blocks.append(.list(ordered: ordered, start: startNumber, items: items.map(parseInline)))
    return max(index, start + 1)
  }

  private static func startsBlock(_ lines: [String], _ index: Int) -> Bool {
    let line = lines[index]
    return matches(fence, line) || matches(heading, line) || matches(rule, line)
      || matches(listItem, line) || matches(quote, line) || isTableStart(lines, index)
  }

  private static func isTableStart(_ lines: [String], _ index: Int) -> Bool {
    lines[index].contains("|") && index + 1 < lines.count && matches(tableDivider, lines[index + 1])
  }

  private static func regex(_ pattern: String) -> NSRegularExpression {
    try! NSRegularExpression(pattern: pattern)
  }

  private static func matches(_ regex: NSRegularExpression, _ line: String) -> Bool {
    regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil
  }

  /// Capture groups after the whole match, nil where a group didn't take part.
  private static func match(_ regex: NSRegularExpression, _ line: String) -> [String?]? {
    guard let result = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line))
    else {
      return nil
    }
    return (1..<result.numberOfRanges).map { index in
      Range(result.range(at: index), in: line).map { String(line[$0]) }
    }
  }
}
