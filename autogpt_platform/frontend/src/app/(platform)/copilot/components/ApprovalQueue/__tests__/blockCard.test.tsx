import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { mail, SESSION_EXEC } from "./fixtures";

function serve(schemaCalls: string[]) {
  server.use(
    http.get(`*/api/review/execution/${SESSION_EXEC}`, () =>
      HttpResponse.json([mail()]),
    ),
    http.get("*/api/builder/blocks/batch", ({ request }) => {
      schemaCalls.push(new URL(request.url).search);
      return HttpResponse.json([
        {
          id: "b-gmail",
          name: "GmailSendBlock",
          inputSchema: {
            type: "object",
            required: ["subject", "to"],
            properties: {
              to: { type: "array", title: "Recipients" },
              subject: { type: "string", title: "Subject line" },
              body: { type: "string", title: "Message" },
            },
          },
        },
      ]);
    }),
  );
}

test("a block's card names the block and labels its inputs from the block's schema", async () => {
  const schemaCalls: string[] = [];
  serve(schemaCalls);
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews graphExecId={SESSION_EXEC} />
    </CopilotChatActionsProvider>,
  );

  expect(
    await screen.findByRole("heading", { name: /Run Gmail Send/ }),
  ).toBeDefined();
  expect(await screen.findByText("Recipients")).toBeDefined();
  expect(screen.getByText("Subject line")).toBeDefined();
  expect(screen.getByText("Message")).toBeDefined();
  expect(screen.getByText("Can't be undone")).toBeDefined();
  expect(schemaCalls.join()).toContain("b-gmail");
  // A key the schema does not list still shows, under the server's label.
  expect(screen.getByText("Account")).toBeDefined();
});
