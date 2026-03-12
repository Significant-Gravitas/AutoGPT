# RecipeSave

A native iOS app (Swift + SwiftUI) that lets you save recipes found on Instagram by pasting a post URL.

## Features

- **Paste any Instagram URL** → the app scrapes the caption and uses Claude AI to extract a structured recipe
- **Intelligent fallback**: if no recipe is found in the caption, the app analyzes the post thumbnail via Claude Vision
- **SwiftData persistence** — all recipes stored fully on-device, no cloud backend
- **Search** recipes by title or ingredient
- **Tag filter chips** on the home screen
- **Edit notes and tags** after saving
- Clean, minimal UI optimised for reading recipes

## AI Service

RecipeSave uses the **Claude API (claude-3-5-haiku-20241022)** for recipe extraction. Claude was chosen because:
- Its tool-use API produces reliable, schema-enforced JSON output
- It handles the noisy, emoji-heavy text of Instagram captions gracefully
- It supports vision — the same API analyses thumbnail images as a fallback
- Fast response times (<3 s) and low cost per extraction

## Setup

### 1. Requirements
- Xcode 15+ (iOS 17 SDK)
- An active [Anthropic API key](https://console.anthropic.com)

### 2. Clone and open
```bash
git clone <repo-url>
open RecipeSave/RecipeSave.xcodeproj
```

### 3. Add your Claude API key
Open `RecipeSave/Info.plist` and replace the placeholder:
```xml
<key>CLAUDE_API_KEY</key>
<string>YOUR_CLAUDE_API_KEY_HERE</string>  <!-- replace this -->
```

> **Security note:** For production, move the key to a `.xcconfig` file tracked in `.gitignore` rather than committing it to `Info.plist`.

### 4. Build & run
Select an iOS 17 simulator and press **⌘R**.

## Project Structure

```
RecipeSave/
├── RecipeSave.xcodeproj/
├── RecipeSave/
│   ├── RecipeSaveApp.swift          # @main, SwiftData container
│   ├── Models/
│   │   └── Recipe.swift             # @Model — SwiftData entity
│   ├── ViewModels/
│   │   ├── RecipeListViewModel.swift # Search + tag filter logic
│   │   ├── AddRecipeViewModel.swift  # Extraction pipeline
│   │   └── RecipeDetailViewModel.swift # Edit notes/tags
│   ├── Views/
│   │   ├── HomeView.swift           # Feed, search bar, tag chips
│   │   ├── AddRecipeView.swift      # URL input + loading state
│   │   ├── RecipeDetailView.swift   # Full recipe display
│   │   └── EditRecipeView.swift     # Edit notes and tags
│   ├── Components/
│   │   ├── RecipeCardView.swift     # List row card
│   │   └── TagChipView.swift        # Reusable pill chip
│   ├── Services/
│   │   ├── InstagramScraperService.swift   # Multi-tier Instagram scraping
│   │   └── ClaudeRecipeExtractorService.swift # Claude API integration
│   └── Info.plist
└── RecipeSaveTests/
    └── RecipeSaveTests.swift        # Unit tests for ViewModels
```

## Architecture

**MVVM** with SwiftData:

```
View ──observes──▶ ViewModel ──calls──▶ Service
                       │
                  ModelContext (SwiftData)
```

All network and AI calls use `async/await`. ViewModels are annotated `@MainActor` and use the `@Observable` macro.

## Instagram Scraping Strategy

Instagram heavily restricts API access. The app uses a tiered approach:

| Tier | Method | Data |
|------|--------|------|
| 1 | `api.instagram.com/oembed` | Thumbnail URL, brief title |
| 2 | HTML fetch (iPhone User-Agent) + Open Graph `og:description` | Full caption |
| 3 | Graceful degradation | Claude analyses thumbnail image |

If all tiers fail, the user sees a clear error message.

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Invalid URL format | Inline hint before any network call |
| Instagram fetch fails | All tiers attempted silently; error shown only if all fail |
| No recipe detected | Alert: "No recipe was detected in this post" |
| Network timeout (15 s) | Alert with description |
| Claude API error / rate limit | Human-readable message from API error body |
| Missing API key | Alert with setup instructions on first use |
