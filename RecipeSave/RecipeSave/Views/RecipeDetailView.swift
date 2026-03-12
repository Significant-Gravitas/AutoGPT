import SwiftUI
import SwiftData

struct RecipeDetailView: View {
    @Environment(\.modelContext) private var context
    let recipe: Recipe

    @State private var showEdit = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Thumbnail
                thumbnailView
                    .frame(maxWidth: .infinity)
                    .frame(height: 280)
                    .clipped()

                VStack(alignment: .leading, spacing: 24) {
                    // Title
                    Text(recipe.title)
                        .font(.title2)
                        .fontWeight(.bold)
                        .fixedSize(horizontal: false, vertical: true)

                    // Tags
                    if !recipe.tags.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(recipe.tags, id: \.self) { tag in
                                    TagChipView(tag: tag)
                                }
                            }
                        }
                    }

                    // Ingredients
                    if !recipe.ingredients.isEmpty {
                        recipeSection(title: "Ingredients") {
                            ForEach(recipe.ingredients, id: \.self) { ingredient in
                                HStack(alignment: .top, spacing: 10) {
                                    Circle()
                                        .frame(width: 6, height: 6)
                                        .foregroundStyle(Color.accentColor)
                                        .padding(.top, 6)
                                    Text(ingredient)
                                        .font(.body)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                    }

                    // Instructions
                    if !recipe.instructions.isEmpty {
                        recipeSection(title: "Instructions") {
                            ForEach(Array(recipe.instructions.enumerated()), id: \.offset) { index, step in
                                HStack(alignment: .top, spacing: 12) {
                                    Text("\(index + 1)")
                                        .font(.caption)
                                        .fontWeight(.bold)
                                        .foregroundStyle(.white)
                                        .frame(width: 24, height: 24)
                                        .background(Color.accentColor)
                                        .clipShape(Circle())
                                        .padding(.top, 1)
                                    Text(step)
                                        .font(.body)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                    }

                    // Notes
                    if !recipe.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        recipeSection(title: "Notes") {
                            Text(recipe.notes)
                                .font(.body)
                                .foregroundStyle(.secondary)
                        }
                    }

                    // Source link
                    if let url = URL(string: recipe.sourceURL) {
                        Link(destination: url) {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.up.right.square")
                                Text("View on Instagram")
                            }
                            .font(.subheadline)
                            .foregroundStyle(Color.accentColor)
                        }
                    }

                    // Date saved
                    Text("Saved \(recipe.dateAdded.formatted(date: .abbreviated, time: .omitted))")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .padding(20)
            }
        }
        .ignoresSafeArea(edges: .top)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Edit") { showEdit = true }
            }
        }
        .sheet(isPresented: $showEdit) {
            EditRecipeView(recipe: recipe)
        }
    }

    @ViewBuilder
    private var thumbnailView: some View {
        if let data = recipe.thumbnailData, let uiImage = UIImage(data: data) {
            Image(uiImage: uiImage)
                .resizable()
                .scaledToFill()
        } else {
            ZStack {
                Color(.secondarySystemBackground)
                Image(systemName: "fork.knife")
                    .font(.system(size: 60))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private func recipeSection<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
                .fontWeight(.semibold)
            content()
        }
    }
}
