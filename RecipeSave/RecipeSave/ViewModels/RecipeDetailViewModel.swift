import Foundation
import SwiftData

@MainActor
@Observable
final class RecipeDetailViewModel {

    var recipe: Recipe
    var editedNotes: String
    var editedTags: [String]
    var newTagInput: String = ""

    init(recipe: Recipe) {
        self.recipe = recipe
        self.editedNotes = recipe.notes
        self.editedTags = recipe.tags
    }

    func addTag() {
        let tag = newTagInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !tag.isEmpty, !editedTags.contains(tag) else {
            newTagInput = ""
            return
        }
        editedTags.append(tag)
        newTagInput = ""
    }

    func removeTag(_ tag: String) {
        editedTags.removeAll { $0 == tag }
    }

    func save(context: ModelContext) {
        recipe.notes = editedNotes
        recipe.tags = editedTags
        try? context.save()
    }
}
