import SwiftUI
import SwiftData

struct HomeView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Recipe.dateAdded, order: .reverse) private var recipes: [Recipe]

    @State private var viewModel = RecipeListViewModel()
    @State private var showAddRecipe = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Tag filter chips
                if !viewModel.allTags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(viewModel.allTags, id: \.self) { tag in
                                TagChipView(
                                    tag: tag,
                                    isSelected: viewModel.selectedTags.contains(tag),
                                    onTap: { viewModel.toggleTag(tag) }
                                )
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 10)
                    }
                    Divider()
                }

                // Recipe list
                let filtered = viewModel.filtered(recipes)

                if filtered.isEmpty {
                    emptyStateView
                } else {
                    List(filtered) { recipe in
                        NavigationLink(destination: RecipeDetailView(recipe: recipe)) {
                            RecipeCardView(recipe: recipe)
                        }
                        .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("RecipeSave")
            .searchable(text: $viewModel.searchText, prompt: "Search recipes or ingredients")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: { showAddRecipe = true }) {
                        Image(systemName: "plus")
                            .fontWeight(.semibold)
                    }
                }
            }
            .sheet(isPresented: $showAddRecipe) {
                AddRecipeView()
            }
            .onChange(of: recipes) { _, new in
                viewModel.update(recipes: new)
            }
            .onAppear {
                viewModel.update(recipes: recipes)
            }
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "fork.knife.circle")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
            if viewModel.searchText.isEmpty && viewModel.selectedTags.isEmpty {
                Text("No recipes yet")
                    .font(.title3)
                    .fontWeight(.semibold)
                Text("Tap + to save a recipe from Instagram.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            } else {
                Text("No results")
                    .font(.title3)
                    .fontWeight(.semibold)
                Text("Try a different search or remove filters.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 40)
        .frame(maxWidth: .infinity)
    }
}
