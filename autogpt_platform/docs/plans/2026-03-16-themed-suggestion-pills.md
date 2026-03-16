# Themed Suggestion Pills — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace flat suggestion pills with 4 themed pill groups (Learn, Create, Automate, Organize) that open popovers with 5 personalized prompts each.

**Architecture:** Backend generates grouped prompts as `dict[str, list[str]]` keyed by theme name. API serves them as `{ themes: [{ name, prompts }] }`. Frontend renders theme pills with shadcn Popover, falling back to hardcoded defaults when no personalized data exists.

**Tech Stack:** Python/Pydantic (backend models), FastAPI (API), Next.js/React (frontend), shadcn Popover, Phosphor Icons.

**Design doc:** `docs/plans/2026-03-16-themed-suggestion-pills-design.md`

---

### Task 1: Update backend data model — `understanding.py`

**Files:**
- Modify: `backend/backend/data/understanding.py:89-92` (BusinessUnderstandingInput.suggested_prompts)
- Modify: `backend/backend/data/understanding.py:130-131` (BusinessUnderstanding.suggested_prompts)
- Modify: `backend/backend/data/understanding.py:160` (from_db — suggested_prompts deserialization)
- Modify: `backend/backend/data/understanding.py:225-229` (merge logic)
- Test: `backend/backend/data/understanding_test.py`

**Step 1: Write failing tests**

Add these tests to `backend/backend/data/understanding_test.py`:

```python
# ─── themed suggested_prompts ─────────────────────────────────────────

THEME_KEYS = ["Learn", "Create", "Automate", "Organize"]


def test_merge_themed_prompts_overwrites_existing():
    """New themed suggested_prompts should fully replace existing ones."""
    existing = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": {
            "Learn": ["Old learn prompt"],
            "Create": ["Old create prompt"],
        },
    }
    new_prompts = {
        "Learn": ["New learn 1", "New learn 2"],
        "Automate": ["New auto 1"],
    }
    input_data = _make_input(suggested_prompts=new_prompts)

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == new_prompts


def test_merge_themed_prompts_none_preserves_existing():
    """When input has suggested_prompts=None, existing themed prompts are preserved."""
    existing_prompts = {"Learn": ["Keep me"], "Create": ["Keep me too"]}
    existing = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": existing_prompts,
    }
    input_data = _make_input(industry="Finance")

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == existing_prompts


def test_from_db_themed_prompts():
    """from_db correctly deserializes themed suggested_prompts dict."""
    from unittest.mock import MagicMock

    themed = {"Learn": ["p1"], "Create": ["p2"]}
    record = MagicMock()
    record.id = "test-id"
    record.userId = "user-1"
    record.createdAt = datetime.now(tz=timezone.utc)
    record.updatedAt = datetime.now(tz=timezone.utc)
    record.data = {"business": {"version": 1}, "suggested_prompts": themed}

    result = BusinessUnderstanding.from_db(record)

    assert result.suggested_prompts == themed


def test_from_db_legacy_list_prompts_returns_empty():
    """from_db ignores legacy list[str] suggested_prompts — returns empty dict."""
    from unittest.mock import MagicMock

    record = MagicMock()
    record.id = "test-id"
    record.userId = "user-1"
    record.createdAt = datetime.now(tz=timezone.utc)
    record.updatedAt = datetime.now(tz=timezone.utc)
    record.data = {"business": {"version": 1}, "suggested_prompts": ["old", "flat"]}

    result = BusinessUnderstanding.from_db(record)

    assert result.suggested_prompts == {}


def test_format_understanding_excludes_themed_prompts():
    """Themed suggested_prompts must NOT appear in system prompt."""
    understanding = BusinessUnderstanding(
        id="test-id",
        user_id="user-1",
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
        user_name="Alice",
        industry="Technology",
        suggested_prompts={"Learn": ["Automate reports"], "Create": ["Draft email"]},
    )

    formatted = format_understanding_for_prompt(understanding)

    assert "suggested_prompts" not in formatted
    assert "Automate reports" not in formatted
    assert "Draft email" not in formatted
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && poetry run pytest backend/data/understanding_test.py -xvs`
Expected: FAIL — type mismatches since models still use `list[str]`

**Step 3: Update the models and logic**

In `understanding.py`:

1. Add a helper to deserialize themed prompts from DB (replacing `_json_to_list` for this field):

```python
def _json_to_themed_prompts(value: Any) -> dict[str, list[str]]:
    """Convert Json field to themed prompts dict, ignoring legacy list format."""
    if isinstance(value, dict):
        return {
            k: v for k, v in value.items()
            if isinstance(k, str) and isinstance(v, list)
        }
    return {}
```

2. Change `BusinessUnderstandingInput.suggested_prompts` (line 90-92):

```python
suggested_prompts: Optional[dict[str, list[str]]] = pydantic.Field(
    None, description="LLM-generated suggested prompts grouped by theme"
)
```

3. Change `BusinessUnderstanding.suggested_prompts` (line 131):

```python
suggested_prompts: dict[str, list[str]] = pydantic.Field(default_factory=dict)
```

4. Update `from_db` (line 160):

```python
suggested_prompts=_json_to_themed_prompts(data.get("suggested_prompts")),
```

5. Merge logic (lines 225-229) stays the same — it already does full-replace on the dict.

**Step 4: Run tests to verify they pass**

Run: `cd backend && poetry run pytest backend/data/understanding_test.py -xvs`
Expected: PASS

**Step 5: Remove old tests that assert `list[str]` behavior**

Delete these tests (they're superseded by the new themed ones):
- `test_merge_suggested_prompts_overwrites_existing`
- `test_merge_suggested_prompts_none_preserves_existing`
- `test_merge_suggested_prompts_added_to_empty_data`
- `test_merge_suggested_prompts_empty_list_overwrites`
- `test_format_understanding_excludes_suggested_prompts`

**Step 6: Run full test file**

Run: `cd backend && poetry run pytest backend/data/understanding_test.py -xvs`
Expected: PASS

**Step 7: Commit**

```bash
git add backend/backend/data/understanding.py backend/backend/data/understanding_test.py
git commit -m "feat(backend): change suggested_prompts from list to themed dict"
```

---

### Task 2: Update API response model — `chat/routes.py`

**Files:**
- Modify: `backend/backend/api/features/chat/routes.py:857-884`
- Test: `backend/backend/api/features/chat/routes_test.py:270-310`

**Step 1: Update tests**

Replace the 3 existing suggested-prompts tests in `routes_test.py` with:

```python
def test_suggested_prompts_returns_themes(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with themed prompts gets them back as themes list."""
    mock_understanding = MagicMock()
    mock_understanding.suggested_prompts = {
        "Learn": ["L1", "L2"],
        "Create": ["C1"],
    }
    _mock_get_business_understanding(mocker, return_value=mock_understanding)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    data = response.json()
    assert "themes" in data
    themes_by_name = {t["name"]: t["prompts"] for t in data["themes"]}
    assert themes_by_name["Learn"] == ["L1", "L2"]
    assert themes_by_name["Create"] == ["C1"]


def test_suggested_prompts_no_understanding(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with no understanding gets empty themes list."""
    _mock_get_business_understanding(mocker, return_value=None)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"themes": []}


def test_suggested_prompts_empty_prompts(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with understanding but empty prompts gets empty themes list."""
    mock_understanding = MagicMock()
    mock_understanding.suggested_prompts = {}
    _mock_get_business_understanding(mocker, return_value=mock_understanding)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"themes": []}
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && poetry run pytest backend/api/features/chat/routes_test.py::test_suggested_prompts_returns_themes -xvs`
Expected: FAIL — response still has `prompts` key

**Step 3: Update the route and response model**

In `routes.py`, replace lines 857-884:

```python
class SuggestedTheme(BaseModel):
    """A themed group of suggested prompts."""
    name: str
    prompts: list[str]


class SuggestedPromptsResponse(BaseModel):
    """Response model for user-specific suggested prompts grouped by theme."""
    themes: list[SuggestedTheme]


@router.get(
    "/suggested-prompts",
    dependencies=[Security(auth.requires_user)],
)
async def get_suggested_prompts(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> SuggestedPromptsResponse:
    """
    Get LLM-generated suggested prompts grouped by theme.

    Returns personalized quick-action prompts based on the user's
    business understanding. Returns empty themes list if no custom
    prompts are available.
    """
    understanding = await get_business_understanding(user_id)
    if understanding is None or not understanding.suggested_prompts:
        return SuggestedPromptsResponse(themes=[])

    themes = [
        SuggestedTheme(name=name, prompts=prompts)
        for name, prompts in understanding.suggested_prompts.items()
    ]
    return SuggestedPromptsResponse(themes=themes)
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && poetry run pytest backend/api/features/chat/routes_test.py -xvs -k suggested`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/backend/api/features/chat/routes.py backend/backend/api/features/chat/routes_test.py
git commit -m "feat(backend): serve suggested prompts grouped by theme"
```

---

### Task 3: Update Tally extraction — `tally.py`

**Files:**
- Modify: `backend/backend/data/tally.py:313-401` (prompt + validation)
- Test: `backend/backend/data/tally_test.py:400-471`

**Step 1: Define theme constants**

Add to the top of `tally.py` (after imports):

```python
SUGGESTION_THEMES = ["Learn", "Create", "Automate", "Organize"]
PROMPTS_PER_THEME = 5
```

**Step 2: Update tests**

Replace `test_extract_business_understanding_from_tally_success` and `test_extract_business_understanding_from_tally_filters_long_prompts` in `tally_test.py`:

```python
@pytest.mark.asyncio
async def test_extract_business_understanding_themed_prompts():
    """Happy path: LLM returns themed prompts as dict."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "user_name": "Alice",
            "business_name": "Acme Corp",
            "suggested_prompts": {
                "Learn": ["Learn 1", "Learn 2", "Learn 3", "Learn 4", "Learn 5"],
                "Create": ["Create 1", "Create 2", "Create 3", "Create 4", "Create 5"],
                "Automate": ["Auto 1", "Auto 2", "Auto 3", "Auto 4", "Auto 5"],
                "Organize": ["Org 1", "Org 2", "Org 3", "Org 4", "Org 5"],
            },
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("backend.data.tally.AsyncOpenAI", return_value=mock_client):
        result = await extract_business_understanding_from_tally("Q: Name?\nA: Alice")

    assert result.user_name == "Alice"
    assert len(result.suggested_prompts) == 4
    assert len(result.suggested_prompts["Learn"]) == 5


@pytest.mark.asyncio
async def test_extract_themed_prompts_filters_long_and_unknown_keys():
    """Long prompts are filtered, unknown keys are dropped, each theme capped at 5."""
    long_prompt = " ".join(["word"] * 21)
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "user_name": "Alice",
            "suggested_prompts": {
                "Learn": [long_prompt, "Valid learn 1", "Valid learn 2"],
                "UnknownTheme": ["Should be dropped"],
                "Automate": ["A1", "A2", "A3", "A4", "A5", "A6"],
            },
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("backend.data.tally.AsyncOpenAI", return_value=mock_client):
        result = await extract_business_understanding_from_tally("Q: Name?\nA: Alice")

    # Unknown key dropped
    assert "UnknownTheme" not in result.suggested_prompts
    # Long prompt filtered
    assert result.suggested_prompts["Learn"] == ["Valid learn 1", "Valid learn 2"]
    # Capped at 5
    assert result.suggested_prompts["Automate"] == ["A1", "A2", "A3", "A4", "A5"]
```

Also update the `test_populate_understanding_from_tally_success` test (line 287):

```python
mock_input.suggested_prompts = {"Learn": ["P1"], "Create": ["P2"]}
```

**Step 3: Run tests to verify they fail**

Run: `cd backend && poetry run pytest backend/data/tally_test.py -xvs -k "themed_prompts or filters_long_and_unknown"  `
Expected: FAIL

**Step 4: Update the extraction prompt and validation**

In `tally.py`, update `_EXTRACTION_PROMPT` (line 334-336) — replace the `suggested_prompts` field description:

```
- suggested_prompts (object with keys "Learn", "Create", "Automate", "Organize"): for each key, \
provide a list of 5 short action prompts (each under 20 words) that would help this person. \
"Learn" = questions about AutoGPT features; "Create" = content/document generation tasks; \
"Automate" = recurring workflow automation ideas; "Organize" = structuring/prioritizing tasks. \
Should be specific to their industry, role, and pain points; actionable and conversational in tone.
```

Update the validation block (lines 384-401):

```python
# Validate suggested_prompts: themed dict, filter >20 words, cap at 5 per theme
raw_prompts = cleaned.get("suggested_prompts", {})
if isinstance(raw_prompts, dict):
    themed: dict[str, list[str]] = {}
    for theme in SUGGESTION_THEMES:
        theme_prompts = raw_prompts.get(theme, [])
        if not isinstance(theme_prompts, list):
            continue
        valid = [
            p.strip()
            for p in theme_prompts
            if isinstance(p, str) and len(p.strip().split()) <= 20
        ]
        if valid:
            themed[theme] = valid[:PROMPTS_PER_THEME]
    if themed:
        cleaned["suggested_prompts"] = themed
    else:
        cleaned.pop("suggested_prompts", None)
else:
    cleaned.pop("suggested_prompts", None)
```

**Step 5: Run tests to verify they pass**

Run: `cd backend && poetry run pytest backend/data/tally_test.py -xvs`
Expected: PASS

**Step 6: Commit**

```bash
git add backend/backend/data/tally.py backend/backend/data/tally_test.py
git commit -m "feat(backend): generate themed suggested prompts from Tally extraction"
```

---

### Task 4: Run full backend checks

**Step 1: Format and lint**

Run: `cd backend && poetry run format && poetry run lint`
Expected: Clean

**Step 2: Run all affected tests**

Run: `cd backend && poetry run pytest backend/data/understanding_test.py backend/data/tally_test.py backend/api/features/chat/routes_test.py -xvs`
Expected: All PASS

**Step 3: Commit any formatting fixes**

```bash
git add -u && git commit -m "style(backend): format"
```

---

### Task 5: Update frontend defaults — `EmptySession/helpers.ts`

**Files:**
- Modify: `frontend/src/app/(platform)/copilot/components/EmptySession/helpers.ts`

**Step 1: Replace quick actions with themed defaults**

Replace `DEFAULT_QUICK_ACTIONS` and `getQuickActions` in `helpers.ts`:

```typescript
export interface SuggestionTheme {
  name: string;
  prompts: string[];
}

export const DEFAULT_THEMES: SuggestionTheme[] = [
  {
    name: "Learn",
    prompts: [
      "What can AutoGPT do for me?",
      "Show me how agents work",
      "What integrations are available?",
      "How do I schedule an agent?",
      "What are the most popular agents?",
    ],
  },
  {
    name: "Create",
    prompts: [
      "Draft a weekly status report",
      "Generate social media posts for my business",
      "Create a competitive analysis summary",
      "Write onboarding emails for new hires",
      "Build a content calendar for next month",
    ],
  },
  {
    name: "Automate",
    prompts: [
      "Monitor my competitors' websites for changes",
      "Send me a daily news digest on my industry",
      "Auto-reply to common customer questions",
      "Track price changes on products I sell",
      "Summarize my emails every morning",
    ],
  },
  {
    name: "Organize",
    prompts: [
      "Sort my bookmarks into categories",
      "Create a project timeline from my notes",
      "Prioritize my task list by urgency",
      "Build a decision matrix for vendor selection",
      "Organize my meeting notes into action items",
    ],
  },
];

export function getSuggestionThemes(
  apiThemes?: SuggestionTheme[],
): SuggestionTheme[] {
  if (apiThemes && apiThemes.length > 0) {
    return apiThemes;
  }
  return DEFAULT_THEMES;
}
```

**Step 2: Commit**

```bash
git add frontend/src/app/\(platform\)/copilot/components/EmptySession/helpers.ts
git commit -m "feat(frontend): add themed suggestion defaults and helper"
```

---

### Task 6: Create SuggestionThemes component

**Files:**
- Create: `frontend/src/app/(platform)/copilot/components/EmptySession/components/SuggestionThemes/SuggestionThemes.tsx`

**Step 1: Create the component**

```tsx
"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/atoms/Button/Button";
import {
  BookOpenIcon,
  PaintBrushIcon,
  LightningIcon,
  ListChecksIcon,
  SpinnerGapIcon,
} from "@phosphor-icons/react";
import { useState } from "react";
import type { SuggestionTheme } from "../../helpers";

const THEME_ICONS: Record<string, typeof BookOpenIcon> = {
  Learn: BookOpenIcon,
  Create: PaintBrushIcon,
  Automate: LightningIcon,
  Organize: ListChecksIcon,
};

interface Props {
  themes: SuggestionTheme[];
  onSend: (prompt: string) => void | Promise<void>;
  disabled?: boolean;
}

export function SuggestionThemes({ themes, onSend, disabled }: Props) {
  const [openTheme, setOpenTheme] = useState<string | null>(null);
  const [loadingPrompt, setLoadingPrompt] = useState<string | null>(null);

  async function handlePromptClick(prompt: string) {
    if (disabled || loadingPrompt) return;
    setLoadingPrompt(prompt);
    try {
      await onSend(prompt);
    } finally {
      setLoadingPrompt(null);
      setOpenTheme(null);
    }
  }

  return (
    <div className="flex flex-wrap items-center justify-center gap-3">
      {themes.map((theme) => {
        const Icon = THEME_ICONS[theme.name];
        return (
          <Popover
            key={theme.name}
            open={openTheme === theme.name}
            onOpenChange={(open) => setOpenTheme(open ? theme.name : null)}
          >
            <PopoverTrigger asChild>
              <Button
                type="button"
                variant="outline"
                size="small"
                disabled={disabled}
                className="shrink-0 gap-2 border-zinc-300 px-3 py-2 text-[.9rem] text-zinc-600"
              >
                {Icon && <Icon size={16} weight="regular" />}
                {theme.name}
              </Button>
            </PopoverTrigger>
            <PopoverContent
              align="center"
              className="w-80 p-2"
              onOpenAutoFocus={(e) => e.preventDefault()}
            >
              <ul className="grid gap-0.5">
                {theme.prompts.map((prompt) => (
                  <li key={prompt}>
                    <button
                      type="button"
                      disabled={loadingPrompt !== null}
                      onClick={() => void handlePromptClick(prompt)}
                      className="w-full rounded-md px-3 py-2 text-left text-sm text-zinc-700 transition-colors hover:bg-zinc-100 disabled:opacity-50"
                    >
                      {loadingPrompt === prompt ? (
                        <span className="flex items-center gap-2">
                          <SpinnerGapIcon
                            className="h-4 w-4 animate-spin"
                            weight="bold"
                          />
                          {prompt}
                        </span>
                      ) : (
                        prompt
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </PopoverContent>
          </Popover>
        );
      })}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/app/\(platform\)/copilot/components/EmptySession/components/SuggestionThemes/SuggestionThemes.tsx
git commit -m "feat(frontend): add SuggestionThemes popover component"
```

---

### Task 7: Update EmptySession to use SuggestionThemes

**Files:**
- Modify: `frontend/src/app/(platform)/copilot/components/EmptySession/EmptySession.tsx`

**Step 1: Replace the flat pills with SuggestionThemes**

Key changes to `EmptySession.tsx`:
1. Import `getSuggestionThemes` instead of `getQuickActions` from helpers
2. Import `SuggestionThemes` component
3. Parse API response into themed format
4. Replace the pills `<div>` with `<SuggestionThemes>`
5. Remove `loadingAction` state (now handled inside SuggestionThemes)
6. Remove `handleQuickActionClick` (now handled inside SuggestionThemes)

Updated component:

```tsx
"use client";

import { useGetV2GetSuggestedPrompts } from "@/app/api/__generated__/endpoints/chat/chat";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { getGreetingName, getInputPlaceholder, getSuggestionThemes } from "./helpers";
import { SuggestionThemes } from "./components/SuggestionThemes/SuggestionThemes";

interface Props {
  inputLayoutId: string;
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (message: string, files?: File[]) => void | Promise<void>;
  isUploadingFiles?: boolean;
  droppedFiles?: File[];
  onDroppedFilesConsumed?: () => void;
}

export function EmptySession({
  inputLayoutId,
  isCreatingSession,
  onSend,
  isUploadingFiles,
  droppedFiles,
  onDroppedFilesConsumed,
}: Props) {
  const { user } = useSupabase();
  const greetingName = getGreetingName(user);

  const { data: suggestedPromptsResponse, isLoading: isLoadingPrompts } =
    useGetV2GetSuggestedPrompts({
      query: { staleTime: Infinity },
    });
  const apiThemes =
    suggestedPromptsResponse?.status === 200
      ? suggestedPromptsResponse.data.themes
      : undefined;
  const themes = getSuggestionThemes(apiThemes);

  const [inputPlaceholder, setInputPlaceholder] = useState(
    getInputPlaceholder(),
  );

  useEffect(() => {
    function update() {
      setInputPlaceholder(getInputPlaceholder(window.innerWidth));
    }
    const mq500 = window.matchMedia("(min-width: 500px)");
    const mq1081 = window.matchMedia("(min-width: 1081px)");
    update();
    mq500.addEventListener("change", update);
    mq1081.addEventListener("change", update);
    return () => {
      mq500.removeEventListener("change", update);
      mq1081.removeEventListener("change", update);
    };
  }, []);

  return (
    <div className="flex h-full flex-1 items-center justify-center overflow-y-auto bg-[#f8f8f9] px-0 py-5 md:px-6 md:py-10">
      <motion.div
        className="w-full max-w-[52rem] text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto max-w-[52rem]">
          <Text variant="h3" className="mb-1 !text-[1.375rem] text-zinc-700">
            Hey, <span className="text-violet-600">{greetingName}</span>
          </Text>
          <Text variant="h3" className="mb-8 !font-normal">
            Tell me about your work — I&apos;ll find what to automate.
          </Text>

          <div className="mb-6">
            <motion.div
              layoutId={inputLayoutId}
              transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
              className="w-full px-2"
            >
              <ChatInput
                inputId="chat-input-empty"
                onSend={onSend}
                disabled={isCreatingSession}
                isUploadingFiles={isUploadingFiles}
                placeholder={inputPlaceholder}
                className="w-full"
                droppedFiles={droppedFiles}
                onDroppedFilesConsumed={onDroppedFilesConsumed}
              />
            </motion.div>
          </div>
        </div>

        {isLoadingPrompts ? (
          <div className="flex flex-wrap items-center justify-center gap-3">
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-10 w-28 shrink-0 rounded-full" />
            ))}
          </div>
        ) : (
          <SuggestionThemes
            themes={themes}
            onSend={onSend}
            disabled={isCreatingSession}
          />
        )}
      </motion.div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/app/\(platform\)/copilot/components/EmptySession/EmptySession.tsx
git commit -m "feat(frontend): wire SuggestionThemes into EmptySession"
```

---

### Task 8: Regenerate API client and run frontend checks

**Step 1: Regenerate OpenAPI spec and frontend client**

This requires the backend running. If the backend can be started locally:

Run: `cd frontend && pnpm generate:api`

If not available locally, manually update the OpenAPI spec to match the new response model and regenerate.

**Step 2: Run frontend checks**

Run: `cd frontend && pnpm format && pnpm lint && pnpm types`
Expected: Clean (or only pre-existing errors in unrelated files)

**Step 3: Commit**

```bash
git add frontend/src/app/api/openapi.json frontend/src/app/api/__generated__/
git commit -m "chore(frontend): regenerate API client for themed suggestions"
```

---

### Task 9: Final verification and push

**Step 1: Run all backend tests**

Run: `cd backend && poetry run pytest backend/data/understanding_test.py backend/data/tally_test.py backend/api/features/chat/routes_test.py -v`
Expected: All PASS

**Step 2: Run frontend checks**

Run: `cd frontend && pnpm format && pnpm lint && pnpm types`
Expected: Clean

**Step 3: Push**

Run: `git push -u origin lluisagusti/secrt-2037-replace-suggestion-pills-with-animated-hint-text-rotation`
