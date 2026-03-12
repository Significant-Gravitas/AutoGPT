import Foundation
import SwiftData

@MainActor
@Observable
final class AddRecipeViewModel {

    var urlText: String = ""
    var isLoading: Bool = false
    var errorMessage: String? = nil
    var didSaveSuccessfully: Bool = false

    private let scraper = InstagramScraperService()
    private let extractor = ClaudeRecipeExtractorService()

    func extractAndSave(context: ModelContext) async {
        let raw = urlText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else {
            errorMessage = "Please paste an Instagram URL."
            return
        }
        guard URL(string: raw) != nil else {
            errorMessage = "The URL you entered doesn't look valid."
            return
        }

        isLoading = true
        errorMessage = nil

        do {
            // Step 1: Scrape the Instagram post
            let post = try await scraper.scrape(url: raw)

            // Step 2: Extract structured recipe via Claude
            let recipeData = try await extractor.extract(caption: post.caption, thumbnailURL: post.thumbnailURL)

            // Step 3: Download thumbnail data (best-effort)
            let thumbnailData = await downloadThumbnail(url: post.thumbnailURL)

            // Step 4: Persist to SwiftData
            let recipe = Recipe(
                title: recipeData.title,
                ingredients: recipeData.ingredients,
                instructions: recipeData.instructions,
                thumbnailData: thumbnailData,
                sourceURL: raw
            )
            context.insert(recipe)
            try context.save()

            didSaveSuccessfully = true
        } catch let error as InstagramScraperError {
            errorMessage = error.localizedDescription
        } catch let error as RecipeExtractionError {
            errorMessage = error.localizedDescription
        } catch {
            errorMessage = "Something went wrong: \(error.localizedDescription)"
        }

        isLoading = false
    }

    func clearError() {
        errorMessage = nil
    }

    // MARK: - Private

    private func downloadThumbnail(url: URL?) async -> Data? {
        guard let url else { return nil }
        return try? await URLSession.shared.data(from: url).0
    }
}
