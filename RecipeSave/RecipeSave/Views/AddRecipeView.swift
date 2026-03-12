import SwiftUI
import SwiftData

struct AddRecipeView: View {
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss

    @State private var viewModel = AddRecipeViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://www.instagram.com/p/…", text: $viewModel.urlText)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                        .disabled(viewModel.isLoading)
                } header: {
                    Text("Instagram URL")
                } footer: {
                    Text("Paste the URL of an Instagram post that contains a recipe.")
                        .foregroundStyle(.secondary)
                }

                Section {
                    Button(action: { Task { await extract() } }) {
                        HStack {
                            Spacer()
                            if viewModel.isLoading {
                                HStack(spacing: 10) {
                                    ProgressView()
                                        .tint(.white)
                                    Text("Extracting recipe…")
                                        .fontWeight(.semibold)
                                }
                            } else {
                                Text("Save Recipe")
                                    .fontWeight(.semibold)
                            }
                            Spacer()
                        }
                    }
                    .disabled(viewModel.isLoading || viewModel.urlText.trimmingCharacters(in: .whitespaces).isEmpty)
                    .listRowBackground(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(
                                viewModel.isLoading || viewModel.urlText.trimmingCharacters(in: .whitespaces).isEmpty
                                ? Color.accentColor.opacity(0.5)
                                : Color.accentColor
                            )
                    )
                    .foregroundStyle(.white)
                }

                if viewModel.isLoading {
                    Section {
                        HStack(spacing: 12) {
                            ProgressView()
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Processing post…")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Text("Scraping caption and extracting recipe via AI")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Add Recipe")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .disabled(viewModel.isLoading)
                }
            }
            .alert("Error", isPresented: Binding(
                get: { viewModel.errorMessage != nil },
                set: { if !$0 { viewModel.clearError() } }
            )) {
                Button("OK") { viewModel.clearError() }
            } message: {
                Text(viewModel.errorMessage ?? "")
            }
            .onChange(of: viewModel.didSaveSuccessfully) { _, success in
                if success { dismiss() }
            }
        }
    }

    private func extract() async {
        await viewModel.extractAndSave(context: context)
    }
}
