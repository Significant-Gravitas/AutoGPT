import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { isIdKey } from "@/components/organisms/ApprovalFields/helpers";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { realCardSchemaHandler, realCards, CHAT_SESSION } from "./fixtures";

const cards = realCards();

test.each(cards.map((card) => [card.story, card] as const))(
  "the real %s card names its subject and labels every input in plain words",
  async (_, card) => {
    server.use(
      http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
        HttpResponse.json([card.review]),
      ),
      realCardSchemaHandler(cards),
    );
    render(
      <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
        <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
      </CopilotChatActionsProvider>,
    );

    const payload = card.review.payload as {
      headline: { ask: string; object: string | null };
      fields: { key: string; label: string }[];
    };
    expect(
      await screen.findByRole("heading", {
        name: new RegExp(
          [payload.headline.ask, payload.headline.object].join(" ").trim(),
        ),
      }),
    ).toBeDefined();
    const card_ = screen.getByRole("region", { name: "Waiting for you" });
    const text = card_.textContent ?? "";
    expect(text).not.toMatch(
      /[a-z][A-Z]\w*Input|ProgrammingLanguage|Literal\[/,
    );
    expect(text).not.toContain("sk-live");
    // Nested values too: a secret reads as hidden, a camelCase key in words.
    expect(text).not.toContain("[redacted]");
    expect(text).not.toContain("mimeType");
    expect(text).not.toContain("MimeType");
    if (card.schema) {
      // Labels come from the schema once it arrives; each shown input has one.
      const shown = payload.fields.find((field) => !isIdKey(field.key));
      if (shown) await screen.findByText(shown.label);
    }
  },
);

test("a code block's step renders as code", async () => {
  const card = cards.find((c) => c.story === "Execute Code Step")!;
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json([card.review]),
    ),
    realCardSchemaHandler(cards),
  );
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
  const code = await screen.findByText(/import pandas as pd/);
  expect(code.closest("pre")).not.toBeNull();
});

test("a held command reads as its text, not as a JSON string", async () => {
  const card = cards.find((c) => c.story === "Sandbox Command")!;
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json([card.review]),
    ),
  );
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
  const code = (await screen.findByText(/post2-hooks-that-convert/)).closest(
    "pre",
  )!;
  const text = code.textContent ?? "";
  expect(text.startsWith("cd /home/user/workspace/blog")).toBe(true);
  expect(text).toContain("'EOF'\n# How to Write");
  expect(text).toContain('"Your posts get likes but no customers."');
  expect(text).not.toContain("\\n");
  expect(text).not.toContain('\\"');
});
