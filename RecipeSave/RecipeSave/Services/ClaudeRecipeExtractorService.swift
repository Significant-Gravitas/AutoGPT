import Foundation

// AI Service Recommendation: Claude API (Anthropic) — claude-3-5-haiku-20241022
//
// Why Claude over OpenAI or other services:
//  1. Structured extraction: Claude's tool-use (function-calling) API produces
//     deterministic JSON output matching an exact schema — ideal for Recipe parsing.
//  2. Noisy text handling: Instagram captions mix emojis, hashtags, and informal
//     language. Claude's instruction-following reliably strips noise and extracts
//     only the recipe content.
//  3. Vision fallback: The same API supports image analysis, letting us pass a
//     thumbnail URL so Claude can read on-screen recipe text from video frames.
//  4. Ecosystem alignment: This app lives in the AutoGPT / Anthropic repo;
//     using Claude keeps dependencies consistent.
//  5. Speed & cost: claude-3-5-haiku-20241022 is fast (<3 s) and cheap per call.
//
// Setup: Add your Claude API key to Info.plist under the key "CLAUDE_API_KEY".
// Get a key at: https://console.anthropic.com

struct RecipeData {
    let title: String
    let ingredients: [String]
    let instructions: [String]
}

enum RecipeExtractionError: LocalizedError {
    case noRecipeFound
    case apiKeyMissing
    case networkError(String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .noRecipeFound:
            return "No recipe was detected in this Instagram post."
        case .apiKeyMissing:
            return "Claude API key is missing. Add CLAUDE_API_KEY to Info.plist."
        case .networkError(let msg):
            return "Network error while extracting recipe: \(msg)"
        case .invalidResponse:
            return "Received an unexpected response from the AI service."
        }
    }
}

actor ClaudeRecipeExtractorService {

    private let apiURL = URL(string: "https://api.anthropic.com/v1/messages")!
    private let model = "claude-3-5-haiku-20241022"

    private var apiKey: String {
        get throws {
            guard let key = Bundle.main.infoDictionary?["CLAUDE_API_KEY"] as? String,
                  !key.isEmpty, key != "YOUR_CLAUDE_API_KEY_HERE" else {
                throw RecipeExtractionError.apiKeyMissing
            }
            return key
        }
    }

    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        return URLSession(configuration: config)
    }()

    // MARK: - Public

    /// Extracts a structured recipe from Instagram caption text.
    /// Falls back to analyzing the thumbnail image if caption yields no recipe.
    func extract(caption: String?, thumbnailURL: URL?) async throws -> RecipeData {
        let key = try apiKey

        // Primary: extract from caption text
        if let caption = caption, !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if let recipe = try? await extractFromText(caption, apiKey: key) {
                return recipe
            }
        }

        // Fallback: analyze thumbnail image for on-screen recipe text
        if let thumbnailURL {
            if let recipe = try? await extractFromImage(url: thumbnailURL, apiKey: key) {
                return recipe
            }
        }

        throw RecipeExtractionError.noRecipeFound
    }

    // MARK: - Private: Text extraction

    private func extractFromText(_ text: String, apiKey: String) async throws -> RecipeData {
        let prompt = """
        The following is text from an Instagram post. Extract the recipe if one is present.
        Return ONLY the JSON tool call — no prose, no markdown fences.
        If there is no recipe, call the tool with {"found": false}.

        Instagram post text:
        \(text)
        """
        return try await callClaudeWithTool(userMessage: prompt, imageURL: nil, apiKey: apiKey)
    }

    // MARK: - Private: Image/vision extraction

    private func extractFromImage(url: URL, apiKey: String) async throws -> RecipeData {
        let prompt = """
        This is a thumbnail or frame from an Instagram cooking video.
        Look for any on-screen recipe text, ingredient lists, or cooking steps visible in the image.
        Extract the recipe if one is present.
        If there is no recipe visible, call the tool with {"found": false}.
        """
        return try await callClaudeWithTool(userMessage: prompt, imageURL: url, apiKey: apiKey)
    }

    // MARK: - Core API call with tool use

    private func callClaudeWithTool(userMessage: String, imageURL: URL?, apiKey: String) async throws -> RecipeData {
        var request = URLRequest(url: apiURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        let body = buildRequestBody(userMessage: userMessage, imageURL: imageURL)
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw RecipeExtractionError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let errMsg = (try? JSONDecoder().decode(ClaudeErrorResponse.self, from: data))?.error.message
                ?? "HTTP \(http.statusCode)"
            throw RecipeExtractionError.networkError(errMsg)
        }

        return try parseResponse(data: data)
    }

    private func buildRequestBody(userMessage: String, imageURL: URL?) -> [String: Any] {
        // Build content blocks
        var contentBlocks: [[String: Any]] = []

        // Optionally prepend an image block for vision analysis
        if let imageURL {
            contentBlocks.append([
                "type": "image",
                "source": [
                    "type": "url",
                    "url": imageURL.absoluteString
                ]
            ])
        }

        contentBlocks.append([
            "type": "text",
            "text": userMessage
        ])

        // Tool schema — Claude will call this with structured recipe data
        let tool: [String: Any] = [
            "name": "extract_recipe",
            "description": "Extract a structured recipe from the provided content.",
            "input_schema": [
                "type": "object",
                "properties": [
                    "found": [
                        "type": "boolean",
                        "description": "Whether a recipe was found in the content."
                    ],
                    "title": [
                        "type": "string",
                        "description": "The recipe title or dish name."
                    ],
                    "ingredients": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "List of ingredients with quantities (e.g. '2 cups flour')."
                    ],
                    "instructions": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Step-by-step cooking instructions in order."
                    ]
                ],
                "required": ["found"]
            ]
        ]

        return [
            "model": model,
            "max_tokens": 1024,
            "tools": [tool],
            "tool_choice": ["type": "any"],
            "messages": [
                [
                    "role": "user",
                    "content": contentBlocks
                ]
            ]
        ]
    }

    private func parseResponse(data: Data) throws -> RecipeData {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = json["content"] as? [[String: Any]] else {
            throw RecipeExtractionError.invalidResponse
        }

        // Find the tool_use block
        for block in content {
            guard block["type"] as? String == "tool_use",
                  let input = block["input"] as? [String: Any] else { continue }

            guard let found = input["found"] as? Bool, found else {
                throw RecipeExtractionError.noRecipeFound
            }

            let title = input["title"] as? String ?? "Untitled Recipe"
            let ingredients = input["ingredients"] as? [String] ?? []
            let instructions = input["instructions"] as? [String] ?? []

            guard !ingredients.isEmpty || !instructions.isEmpty else {
                throw RecipeExtractionError.noRecipeFound
            }

            return RecipeData(title: title, ingredients: ingredients, instructions: instructions)
        }

        throw RecipeExtractionError.noRecipeFound
    }
}

// MARK: - Error response model

private struct ClaudeErrorResponse: Decodable {
    struct APIError: Decodable { let message: String }
    let error: APIError
}
