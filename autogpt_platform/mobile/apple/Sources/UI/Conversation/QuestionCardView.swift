import SwiftUI

/// The native card for clarifying questions. Single, multiple and custom
/// answers are sent as one reply worded exactly as the web client words it.
/// The card only accepts answers while the asking reply is the latest message
/// and nothing is running, so a reloaded or superseded card can't resend.
struct QuestionCardView: View {
  let toolCallID: String
  let questions: QuestionSet
  let pending: Bool
  let onSubmit: (String) -> Void
  let onDetails: () -> Void
  @State private var answers: [String: QuestionAnswer] = [:]
  @State private var submitted = false

  private var editable: Bool { pending && !submitted }

  private var allAnswered: Bool {
    questions.questions.allSatisfy { answers[$0.keyword]?.isAnswered == true }
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Label(pending ? "Needs your input" : "Questions", systemImage: "questionmark.circle")
          .font(.footnote.weight(.semibold))
          .foregroundStyle(.secondary)
        Spacer()
        Button("Details", action: onDetails)
          .font(.footnote)
          .buttonStyle(.borderless)
          .frame(minWidth: 44, minHeight: 44)
      }
      if let message = questions.message {
        Text(message).textSelection(.enabled)
      }
      ForEach(questions.questions) { question in
        QuestionField(question: question, answer: binding(question.keyword), enabled: editable)
      }
      if pending {
        Button {
          guard allAnswered, !submitted else { return }
          submitted = true
          onSubmit(questions.answersMessage(answers))
        } label: {
          Text(submitted ? "Sending…" : "Send Answers").frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .buttonBorderShape(.capsule)
        .controlSize(.large)
        .disabled(!allAnswered || submitted)
        .accessibilityIdentifier("question-submit-\(toolCallID)")
      } else {
        Text("This question isn't waiting for an answer anymore.")
          .font(.footnote)
          .foregroundStyle(.secondary)
      }
    }
    .padding(16)
    .background(
      Color(uiColor: .systemBackground),
      in: RoundedRectangle(cornerRadius: 20, style: .continuous)
    )
    .overlay(
      RoundedRectangle(cornerRadius: 20, style: .continuous)
        .strokeBorder(Brand.border, lineWidth: 1)
    )
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("question-card-\(toolCallID)")
  }

  private func binding(_ keyword: String) -> Binding<QuestionAnswer?> {
    Binding(get: { answers[keyword] }, set: { answers[keyword] = $0 })
  }
}

private struct QuestionField: View {
  let question: ClarifyingQuestion
  @Binding var answer: QuestionAnswer?
  let enabled: Bool
  @FocusState private var customFocused: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(question.question)
        .font(.headline)
        .accessibilityAddTraits(.isHeader)
      if question.allowsMultiple {
        Text("Select all that apply")
          .font(.footnote)
          .foregroundStyle(.secondary)
      }
      ForEach(Array(question.options.enumerated()), id: \.offset) { index, option in
        let selected = isSelected(option)
        Button {
          toggle(option)
        } label: {
          HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(option).foregroundStyle(.primary).multilineTextAlignment(.leading)
            Spacer(minLength: 0)
            Image(systemName: symbol(selected: selected))
              .foregroundStyle(selected ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
              .accessibilityHidden(true)
          }
          .padding(.horizontal, 12)
          .padding(.vertical, 10)
          .frame(minHeight: 44)
          .background(Brand.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
          .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
              .strokeBorder(selected ? Brand.prominent : Color.clear, lineWidth: 1.5)
          )
          .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityHint(Text(question.allowsMultiple ? "Select any that apply" : "Select one"))
        .accessibilityIdentifier("question-option-\(question.keyword)-\(index)")
      }
      TextField(
        placeholder, text: customText, prompt: Text(question.example ?? placeholder),
        axis: .vertical
      )
      .lineLimit(1...4)
      .padding(12)
      .background(Brand.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
      .focused($customFocused)
      .disabled(!enabled)
      .accessibilityIdentifier("question-custom-\(question.keyword)")
    }
    .onChange(of: enabled) { _, enabled in
      if !enabled { customFocused = false }
    }
    .toolbar {
      ToolbarItemGroup(placement: .keyboard) {
        if customFocused {
          Spacer()
          Button("Done") { customFocused = false }
            .accessibilityIdentifier("question-dismiss-keyboard")
        }
      }
    }
  }

  private var placeholder: String {
    question.options.isEmpty ? "Your answer" : "Something else"
  }

  private func symbol(selected: Bool) -> String {
    if question.allowsMultiple { return selected ? "checkmark.square.fill" : "square" }
    return selected ? "checkmark.circle.fill" : "circle"
  }

  private func isSelected(_ option: String) -> Bool {
    switch answer {
    case .text(let text)?: text == option
    case .multiple(let selected, _)?: selected.contains(option)
    case nil: false
    }
  }

  private func toggle(_ option: String) {
    if question.allowsMultiple {
      var selected = multiple.selected
      let custom = multiple.custom
      if let index = selected.firstIndex(of: option) {
        selected.remove(at: index)
      } else {
        selected.append(option)
      }
      answer = .multiple(selected: selected, custom: custom)
    } else {
      answer = .text(option)
    }
  }

  private var multiple: (selected: [String], custom: String) {
    switch answer {
    case .multiple(let selected, let custom)?: (selected, custom)
    case .text(let text)?: ([], text)
    case nil: ([], "")
    }
  }

  /// Typed text: a single answer's text unless it is one of the options, or
  /// a multiple answer's own slot.
  private var customText: Binding<String> {
    Binding(
      get: {
        switch answer {
        case .text(let text)?: question.options.contains(text) ? "" : text
        case .multiple(_, let custom)?: custom
        case nil: ""
        }
      },
      set: { typed in
        if question.allowsMultiple {
          answer = .multiple(selected: multiple.selected, custom: typed)
        } else {
          answer = .text(typed)
        }
      })
  }
}

/// Read-only details for one tool step: the bounded, secret-masked input and
/// result, and a plain statement of what the app can't do with it yet.
struct ToolDetailsView: View {
  let part: ToolPart
  @Environment(\.dismiss) private var dismiss
  @Environment(\.dynamicTypeSize) private var typeSize

  var body: some View {
    let presentation = ToolPresentation(part)
    NavigationStack {
      List {
        Section {
          LabeledContent("Status", value: presentation.statusLabel)
          LabeledContent("Tool", value: part.toolName.isEmpty ? "Unknown" : part.toolName)
          LabeledContent("Result type", value: presentation.responseType ?? "None")
          LabeledContent("Card", value: presentation.family?.label ?? "Not recognized")
        } footer: {
          Text(presentation.coverageNote)
        }
        if let summary = presentation.summary {
          Section("Summary") {
            Text(summary).textSelection(.enabled)
          }
        }
        Section("Input") {
          code(BoundedJSON.prettyPrint(part.input))
        }
        if let output = part.output {
          Section("Result") {
            code(BoundedJSON.prettyPrint(output))
          }
        }
      }
      .navigationTitle(presentation.title)
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") { dismiss() }
        }
      }
    }
    .presentationDetents([.medium, .large])
  }

  private func code(_ text: String) -> some View {
    SelectableText(
      text: NSAttributedString(
        string: text,
        attributes: [
          .font: TextStyleFont.font(.callout, typeSize, monospaced: true),
          .foregroundColor: UIColor.label,
        ]))
  }
}
