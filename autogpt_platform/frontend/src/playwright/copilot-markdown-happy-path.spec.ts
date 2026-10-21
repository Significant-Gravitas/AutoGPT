import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { CopilotPage } from "./pages/copilot.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

const DIAGRAM_REPLY = [
  "Here is how the steps connect:",
  "",
  "```mermaid",
  "graph TD",
  "  Fetch[Fetch rows] --> Clean[Clean rows]",
  "```",
].join("\n");

const FORMULA_REPLY = "The total is $\\sum_{i=1}^{n} x_i$ for every row.";

function historyRow(
  sessionId: string,
  sequence: number,
  role: "user" | "assistant",
  content: string,
) {
  return {
    id: `markdown-e2e-${sequence}`,
    session_id: sessionId,
    role,
    content,
    tool_calls: null,
    tool_call_id: null,
    sequence,
    duration_ms: role === "assistant" ? 1200 : null,
    created_at: new Date(Date.UTC(2026, 0, 5, 9, 0, sequence)).toISOString(),
    metadata: {},
  };
}

test("copilot happy path: a reply's diagram and formula render after their plugins load", async ({
  page,
}) => {
  test.setTimeout(120000);

  const copilotPage = new CopilotPage(page);
  await copilotPage.open();
  const sessionId = await copilotPage.createSessionViaApi();

  // The chat's history comes from the real session, with two finished turns
  // added: one reply holds a mermaid diagram, the other a formula.
  await page.route(
    (url) => url.pathname === `/api/proxy/api/chat/sessions/${sessionId}`,
    async (route) => {
      if (route.request().method() !== "GET") return route.fallback();
      const response = await route.fetch();
      const session = await response.json();
      const messages = [
        historyRow(sessionId, 0, "user", "Draw the cleanup steps"),
        historyRow(sessionId, 1, "assistant", DIAGRAM_REPLY),
        historyRow(sessionId, 2, "user", "And the total?"),
        historyRow(sessionId, 3, "assistant", FORMULA_REPLY),
      ];
      await route.fulfill({
        response,
        json: {
          ...session,
          messages,
          oldest_sequence: 0,
          has_more_messages: false,
        },
      });
    },
  );

  await copilotPage.open(sessionId);
  await copilotPage.waitForChatInput();

  const diagram = page
    .locator('[data-streamdown="mermaid"] svg[id^="mermaid-"]')
    .first();
  await diagram.scrollIntoViewIfNeeded({ timeout: 30000 });
  await expect(
    diagram,
    "the mermaid block must be drawn as an SVG diagram",
  ).toBeVisible({ timeout: 30000 });
  await expect(diagram).toContainText("Fetch rows");
  await expect(diagram).toContainText("Clean rows");

  const formula = page.locator(".katex").first();
  await formula.scrollIntoViewIfNeeded({ timeout: 15000 });
  await expect(
    formula,
    "inline $…$ math must render through KaTeX",
  ).toBeVisible({ timeout: 15000 });
  await expect(
    page.locator('.katex annotation[encoding="application/x-tex"]').first(),
  ).toHaveText("\\sum_{i=1}^{n} x_i");
});
