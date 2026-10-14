import Foundation

/// A clarifying question from `ask_question`
/// (`agent_builder_clarification_needed`), normalized as the web client does.
struct ClarifyingQuestion: Equatable, Sendable, Identifiable {
  let question: String
  let keyword: String
  let example: String?
  let options: [String]
  /// Several of `options` may be picked. Never set without options.
  let allowsMultiple: Bool

  var id: String { keyword }
}

/// One answer: a single pick or typed text, or several picks plus optional text.
enum QuestionAnswer: Equatable, Sendable {
  case text(String)
  case multiple(selected: [String], custom: String)

  /// The answer as its list of picks, blanks dropped; empty means unanswered.
  var picks: [String] {
    let raw: [String]
    switch self {
    case .text(let text): raw = [text]
    case .multiple(let selected, let custom): raw = selected + [custom]
    }
    return raw.map(\.trimmed).filter { !$0.isEmpty }
  }

  var isAnswered: Bool { !picks.isEmpty }

  /// Several picks become a bullet list; one stays inline.
  var formatted: String {
    let picks = self.picks
    return picks.count > 1 ? picks.map { "- \($0)" }.joined(separator: "\n") : picks.first ?? ""
  }
}

struct QuestionSet: Equatable, Sendable {
  let message: String?
  let questions: [ClarifyingQuestion]

  /// A question output, as the web's `isClarificationOutput` recognizes one.
  init?(_ part: ToolPart) {
    guard part.state == .done, var output = part.output else { return nil }
    if let text = output.string { output = JSONValue.parseLoose(text) }
    let isQuestion =
      output["type"]?.string == "agent_builder_clarification_needed" || output["questions"] != nil
    guard isQuestion, let raw = output["questions"]?.array, !raw.isEmpty else { return nil }
    let questions = Self.normalize(raw.filter { $0.object != nil })
    guard !questions.isEmpty else { return nil }
    self.message = output["message"]?.nonEmptyString
    self.questions = questions
  }

  init(message: String?, questions: [ClarifyingQuestion]) {
    self.message = message
    self.questions = questions
  }

  /// The reply the web client sends for the same answers, so the server and
  /// model see one wording whichever client answered.
  func answersMessage(_ answers: [String: QuestionAnswer]) -> String {
    let body = questions.map { question in
      "> \(question.question)\n\n\(answers[question.keyword]?.formatted ?? "")"
    }
    return "**Here are my answers:**\n\n\(body.joined(separator: "\n\n"))\n\nPlease proceed."
  }

  private static func normalize(_ raw: [JSONValue]) -> [ClarifyingQuestion] {
    var seen = Set<String>()
    return raw.enumerated().map { index, item in
      var keyword = item["keyword"]?.string?.trimmed.lowercased() ?? ""
      if keyword.isEmpty { keyword = "question-\(index)" }
      var unique = keyword
      var suffix = 1
      while seen.contains(unique) {
        unique = "\(keyword)-\(suffix)"
        suffix += 1
      }
      seen.insert(unique)
      var options: [String] = []
      for option in item["options"]?.array ?? [] {
        if let text = option.nonEmptyString, !options.contains(text) { options.append(text) }
      }
      return ClarifyingQuestion(
        question: item["question"]?.string ?? "",
        keyword: unique,
        example: item["example"]?.nonEmptyString,
        options: options,
        allowsMultiple: !options.isEmpty && item["allow_multiple"]?.bool == true)
    }
  }
}
