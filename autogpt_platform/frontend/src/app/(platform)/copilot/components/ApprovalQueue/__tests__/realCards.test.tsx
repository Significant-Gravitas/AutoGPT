import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
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
      subject: { name: string };
      fields: { key: string; label: string }[];
    };
    expect(
      await screen.findByRole("heading", {
        name: new RegExp(`Run ${payload.subject.name}`),
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
      await screen.findByText(payload.fields[0].label);
    }
  },
);
