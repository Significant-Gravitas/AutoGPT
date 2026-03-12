import SwiftUI
import SwiftData

@main
struct RecipeSaveApp: App {
    var body: some Scene {
        WindowGroup {
            HomeView()
                .modelContainer(for: Recipe.self)
        }
    }
}
