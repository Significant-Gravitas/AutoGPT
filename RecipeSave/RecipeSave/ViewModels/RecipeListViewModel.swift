import Foundation
import SwiftData
import Combine

@MainActor
@Observable
final class RecipeListViewModel {

    var searchText: String = ""
    var selectedTags: Set<String> = []

    // Derived from all persisted recipes — call update() when the recipe list changes
    private(set) var allTags: [String] = []

    func update(recipes: [Recipe]) {
        let tagSet = recipes.reduce(into: Set<String>()) { $0.formUnion($1.tags) }
        allTags = tagSet.sorted()
    }

    func filtered(_ recipes: [Recipe]) -> [Recipe] {
        recipes.filter { recipe in
            let matchesSearch: Bool = {
                guard !searchText.trimmingCharacters(in: .whitespaces).isEmpty else { return true }
                let query = searchText.lowercased()
                return recipe.title.lowercased().contains(query)
                    || recipe.ingredients.contains { $0.lowercased().contains(query) }
            }()

            let matchesTags: Bool = {
                guard !selectedTags.isEmpty else { return true }
                return !selectedTags.isDisjoint(with: recipe.tags)
            }()

            return matchesSearch && matchesTags
        }
    }

    func toggleTag(_ tag: String) {
        if selectedTags.contains(tag) {
            selectedTags.remove(tag)
        } else {
            selectedTags.insert(tag)
        }
    }
}
