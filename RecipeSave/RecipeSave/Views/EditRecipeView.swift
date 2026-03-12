import SwiftUI
import SwiftData

struct EditRecipeView: View {
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss

    @State private var viewModel: RecipeDetailViewModel

    init(recipe: Recipe) {
        _viewModel = State(initialValue: RecipeDetailViewModel(recipe: recipe))
    }

    var body: some View {
        NavigationStack {
            Form {
                // Notes
                Section {
                    ZStack(alignment: .topLeading) {
                        if viewModel.editedNotes.isEmpty {
                            Text("Add personal notes, substitutions, or tips…")
                                .foregroundStyle(.tertiary)
                                .padding(.top, 8)
                                .padding(.leading, 4)
                        }
                        TextEditor(text: $viewModel.editedNotes)
                            .frame(minHeight: 120)
                    }
                } header: {
                    Text("Notes")
                }

                // Tags
                Section {
                    // Existing tags
                    if !viewModel.editedTags.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(viewModel.editedTags, id: \.self) { tag in
                                    TagChipView(tag: tag, onRemove: {
                                        viewModel.removeTag(tag)
                                    })
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }

                    // Add new tag
                    HStack {
                        TextField("Add a tag (e.g. dinner, vegan)", text: $viewModel.newTagInput)
                            .autocorrectionDisabled()
                            .onSubmit { viewModel.addTag() }

                        if !viewModel.newTagInput.trimmingCharacters(in: .whitespaces).isEmpty {
                            Button(action: viewModel.addTag) {
                                Image(systemName: "plus.circle.fill")
                                    .foregroundStyle(Color.accentColor)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                } header: {
                    Text("Tags")
                } footer: {
                    Text("Tags help you filter your recipes on the home screen.")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Edit Recipe")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        viewModel.save(context: context)
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}
