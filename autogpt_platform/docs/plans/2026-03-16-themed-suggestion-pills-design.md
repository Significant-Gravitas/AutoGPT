# Themed Suggestion Pills

**Date**: 2026-03-16
**Ticket**: SECRT-2037
**Branch**: `lluisagusti/secrt-2037-replace-suggestion-pills-with-animated-hint-text-rotation`

## Summary

Replace the current flat suggestion pills in the CoPilot empty session with themed pill groups. Four fixed themes (Learn, Create, Automate, Organize) are shown as pills. Clicking a theme opens a popover with 5 personalized prompts. Clicking a prompt fires it as a chat message.

## Decisions

- **Fixed themes, dynamic prompts** (Option A) — theme names are always the same; LLM generates prompts per theme based on Tally business understanding
- **Fallback**: hardcoded generic prompts when no Tally data exists
- **Interaction**: Popover anchored to the pill (shadcn Popover), dismisses on outside click
- **5 prompts per theme**, each under 20 words
- **Backend generates grouped data** (Approach 1) — single source of truth

## Data Model

### Backend — `understanding.py`

`suggested_prompts` changes from `list[str]` to `dict[str, list[str]]`:

```python
# Shape
{
  "Learn": ["...", "...", "...", "...", "..."],
  "Create": ["...", "...", "...", "...", "..."],
  "Automate": ["...", "...", "...", "...", "..."],
  "Organize": ["...", "...", "...", "...", "..."],
}
```

- 4 fixed theme keys defined as constants
- `merge_business_understanding_data`: same full-replace strategy
- `from_db()`: if existing data is a `list[str]` (legacy), ignore and fallback to defaults
- `format_understanding_for_prompt()`: continues to exclude `suggested_prompts`

### API — `chat/routes.py`

```python
class SuggestedTheme(BaseModel):
    name: str
    prompts: list[str]

class SuggestedPromptsResponse(BaseModel):
    themes: list[SuggestedTheme]
```

Endpoint stays `GET /suggested-prompts`. Returns 4 themes, each with 5 prompts.

### LLM Prompt — `tally.py`

Updated extraction prompt asks for 5 prompts per theme as a JSON object with the 4 fixed keys. Validation:
- Filter prompts > 20 words
- Ensure each theme has exactly 5 (pad from hardcoded defaults if short)
- Ignore unknown theme keys

## Frontend

### Updated: `EmptySession/`

- Replace flat pills with `SuggestionThemes` component
- `useGetV2GetSuggestedPrompts` consumes new `{ themes: SuggestedTheme[] }` response
- Fallback: hardcoded default themes when API returns empty

### New: `EmptySession/components/SuggestionThemes/`

- `SuggestionThemes.tsx` — 4 theme pills in a centered horizontal row
- Each pill is a shadcn `Popover` trigger
- Popover shows 5 prompts as clickable list items
- Clicking a prompt fires `onSend(prompt)` and closes the popover
- Loading state: 4 skeleton pills
- Each pill gets a small Phosphor icon (BookOpen, PaintBrush, Lightning, ListChecks)

### Interaction Flow

1. User sees 4 theme pills below the input
2. Clicks a pill → popover opens with 5 suggestions
3. Clicks a suggestion → `onSend(prompt)` fires, popover closes
4. Clicks outside → popover dismisses

## Hardcoded Default Themes

| Theme | Prompts |
|-------|---------|
| Learn | "What can AutoGPT do for me?", "Show me how agents work", "What integrations are available?", "How do I schedule an agent?", "What are the most popular agents?" |
| Create | "Draft a weekly status report", "Generate social media posts for my business", "Create a competitive analysis summary", "Write onboarding emails for new hires", "Build a content calendar for next month" |
| Automate | "Monitor my competitors' websites for changes", "Send me a daily news digest on my industry", "Auto-reply to common customer questions", "Track price changes on products I sell", "Summarize my emails every morning" |
| Organize | "Sort my bookmarks into categories", "Create a project timeline from my notes", "Prioritize my task list by urgency", "Build a decision matrix for vendor selection", "Organize my meeting notes into action items" |

## Edge Cases

- **LLM returns < 5 prompts for a theme**: pad with hardcoded defaults for that theme
- **LLM returns unknown theme key**: ignore, only use the 4 fixed keys
- **Legacy `list[str]` in DB**: ignored, fallback to defaults until next Tally extraction regenerates
- **API error / loading**: show 4 skeleton pills, then hardcoded defaults on error

## Files Changed

### Backend
- `backend/data/understanding.py` — model change, merge logic, backward compat
- `backend/data/understanding_test.py` — updated tests
- `backend/data/tally.py` — LLM prompt update, validation
- `backend/data/tally_test.py` — updated tests
- `backend/api/features/chat/routes.py` — response model change
- `backend/api/features/chat/routes_test.py` — updated tests

### Frontend
- `frontend/src/app/(platform)/copilot/components/EmptySession/EmptySession.tsx` — use SuggestionThemes
- `frontend/src/app/(platform)/copilot/components/EmptySession/helpers.ts` — default themes data
- `frontend/src/app/(platform)/copilot/components/EmptySession/components/SuggestionThemes/SuggestionThemes.tsx` — new component
- `frontend/src/app/api/openapi.json` — regenerated
- Generated API types — regenerated via `pnpm generate:api`
