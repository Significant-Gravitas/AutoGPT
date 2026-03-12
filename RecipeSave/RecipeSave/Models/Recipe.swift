import Foundation
import SwiftData

@Model
final class Recipe {
    var id: UUID
    var title: String
    var ingredients: [String]
    var instructions: [String]
    var thumbnailData: Data?
    var sourceURL: String
    var notes: String
    var tags: [String]
    var dateAdded: Date

    init(
        id: UUID = UUID(),
        title: String,
        ingredients: [String],
        instructions: [String],
        thumbnailData: Data? = nil,
        sourceURL: String,
        notes: String = "",
        tags: [String] = [],
        dateAdded: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.ingredients = ingredients
        self.instructions = instructions
        self.thumbnailData = thumbnailData
        self.sourceURL = sourceURL
        self.notes = notes
        self.tags = tags
        self.dateAdded = dateAdded
    }
}
