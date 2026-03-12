import XCTest
@testable import RecipeSave

final class RecipeSaveTests: XCTestCase {

    // MARK: - RecipeListViewModel filter tests

    @MainActor
    func testFilterBySearchText() {
        let vm = RecipeListViewModel()
        let recipes = [
            makeRecipe(title: "Pasta Carbonara", ingredients: ["eggs", "pancetta"]),
            makeRecipe(title: "Avocado Toast", ingredients: ["avocado", "bread"]),
            makeRecipe(title: "Lemon Chicken", ingredients: ["chicken", "lemon"]),
        ]
        vm.searchText = "avo"
        let results = vm.filtered(recipes)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.title, "Avocado Toast")
    }

    @MainActor
    func testFilterByIngredient() {
        let vm = RecipeListViewModel()
        let recipes = [
            makeRecipe(title: "Pancake Stack", ingredients: ["flour", "eggs", "milk"]),
            makeRecipe(title: "Veggie Stir Fry", ingredients: ["tofu", "broccoli"]),
        ]
        vm.searchText = "eggs"
        let results = vm.filtered(recipes)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.title, "Pancake Stack")
    }

    @MainActor
    func testFilterByTag() {
        let vm = RecipeListViewModel()
        let recipes = [
            makeRecipe(title: "Buddha Bowl", tags: ["vegan", "healthy"]),
            makeRecipe(title: "Beef Burger", tags: ["comfort"]),
        ]
        vm.selectedTags = ["vegan"]
        let results = vm.filtered(recipes)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.title, "Buddha Bowl")
    }

    @MainActor
    func testEmptySearchReturnsAll() {
        let vm = RecipeListViewModel()
        let recipes = [
            makeRecipe(title: "Recipe A"),
            makeRecipe(title: "Recipe B"),
            makeRecipe(title: "Recipe C"),
        ]
        let results = vm.filtered(recipes)
        XCTAssertEqual(results.count, 3)
    }

    @MainActor
    func testAllTagsAggregation() {
        let vm = RecipeListViewModel()
        let recipes = [
            makeRecipe(title: "A", tags: ["breakfast", "easy"]),
            makeRecipe(title: "B", tags: ["dinner"]),
            makeRecipe(title: "C", tags: ["easy", "vegan"]),
        ]
        vm.update(recipes: recipes)
        XCTAssertEqual(Set(vm.allTags), ["breakfast", "dinner", "easy", "vegan"])
    }

    // MARK: - RecipeDetailViewModel tests

    @MainActor
    func testAddTag() {
        let recipe = makeRecipe(title: "Test", tags: ["italian"])
        let vm = RecipeDetailViewModel(recipe: recipe)
        vm.newTagInput = "pasta"
        vm.addTag()
        XCTAssertTrue(vm.editedTags.contains("pasta"))
        XCTAssertEqual(vm.newTagInput, "")
    }

    @MainActor
    func testAddDuplicateTagIgnored() {
        let recipe = makeRecipe(title: "Test", tags: ["italian"])
        let vm = RecipeDetailViewModel(recipe: recipe)
        vm.newTagInput = "italian"
        vm.addTag()
        XCTAssertEqual(vm.editedTags.filter { $0 == "italian" }.count, 1)
    }

    @MainActor
    func testRemoveTag() {
        let recipe = makeRecipe(title: "Test", tags: ["vegan", "quick"])
        let vm = RecipeDetailViewModel(recipe: recipe)
        vm.removeTag("vegan")
        XCTAssertFalse(vm.editedTags.contains("vegan"))
        XCTAssertTrue(vm.editedTags.contains("quick"))
    }

    // MARK: - Helpers

    private func makeRecipe(
        title: String,
        ingredients: [String] = [],
        instructions: [String] = [],
        tags: [String] = []
    ) -> Recipe {
        Recipe(
            title: title,
            ingredients: ingredients,
            instructions: instructions,
            sourceURL: "https://www.instagram.com/p/test/",
            tags: tags
        )
    }
}
