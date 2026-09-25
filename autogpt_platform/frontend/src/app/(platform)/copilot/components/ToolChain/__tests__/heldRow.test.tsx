import { expect, test, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import type { HeldOutcome } from "../../ChatMessagesContainer/heldCallRows";
import { HeldOutcomesContext } from "../../ChatMessagesContainer/HeldOutcomesContext";
import { ToolChain } from "../ToolChain";

const HELD: MessagePart = {
  type: "tool-create_folder",
  state: "output-available",
  toolCallId: "call-7",
  input: { name: "Q3 reports" },
  output: {
    type: "approval_required",
    tool_name: "create_folder",
    reason: "Ask First is on.",
    review_id: "copilot-node-gate-create_folder:abc",
    ask: "Create folder",
    object: "Q3 reports",
  },
} as MessagePart;

function chain(outcomes: Map<string, HeldOutcome>) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()}>
      <HeldOutcomesContext.Provider value={outcomes}>
        <ToolChain parts={[HELD]} isStreaming={false} />
      </HeldOutcomesContext.Provider>
    </CopilotChatActionsProvider>
  );
}

test("a held call's row flips from waiting to approved when its late result lands", async () => {
  const { rerender } = render(chain(new Map()));

  expect(await screen.findByText("Waiting for you")).toBeDefined();
  expect(
    screen.getAllByText('Create folder "Q3 reports"').length,
  ).toBeGreaterThan(0);

  rerender(
    chain(
      new Map([
        ["call-7", { outcome: "approved", output: { message: "Created" } }],
      ]),
    ),
  );

  expect(await screen.findByText("Approved")).toBeDefined();
  expect(
    screen.getAllByText('Created folder "Q3 reports"').length,
  ).toBeGreaterThan(0);
  expect(screen.queryByText("Waiting for you")).toBeNull();
});

test("a late result for another call leaves the row waiting", async () => {
  render(chain(new Map([["call-8", { outcome: "approved", output: "done" }]])));
  expect(await screen.findByText("Waiting for you")).toBeDefined();
});

test.each([
  ["rejected", "Rejected", "Didn't create folder"],
  ["expired", "Expired", "Didn't create folder"],
] as const)("a %s call says so on its row", async (outcome, tag, label) => {
  render(chain(new Map([["call-7", { outcome, output: "" }]])));
  expect(await screen.findByText(tag)).toBeDefined();
  expect(screen.getAllByText(new RegExp(label)).length).toBeGreaterThan(0);
});

test("an approved call that failed when it ran shows its error, still marked approved", async () => {
  render(
    chain(
      new Map([
        [
          "call-7",
          {
            outcome: "approved",
            output: { type: "error", message: "Folder exists" },
          },
        ],
      ]),
    ),
  );
  expect(await screen.findByText("Approved")).toBeDefined();
  // The error line under the row, as an unheld call's failure shows it.
  const errorLine = screen
    .getAllByText("Folder exists")
    .find((el) => el.className.includes("text-red"));
  expect(errorLine).toBeDefined();
});

const HELD_READ: MessagePart = {
  type: "tool-web_fetch",
  state: "output-available",
  toolCallId: "call-9",
  input: { url: "docs.northwind.io/billing" },
  output: {
    type: "approval_required",
    tool_name: "web_fetch",
    reason:
      "Content withheld pending your review: web_fetch docs.northwind.io/billing.",
    review_id: "copilot-node-gate-read-web_fetch:abc",
    ask: "Read",
    object: "docs.northwind.io/billing",
  },
} as MessagePart;

function readChain(outcomes: Map<string, HeldOutcome>) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()}>
      <HeldOutcomesContext.Provider value={outcomes}>
        <ToolChain parts={[HELD_READ]} isStreaming={false} />
      </HeldOutcomesContext.Provider>
    </CopilotChatActionsProvider>
  );
}

test("a held read's row reads Held, then Kept out, and keeps naming the read", async () => {
  const { rerender } = render(readChain(new Map()));

  expect(await screen.findByText("Held")).toBeDefined();
  expect(
    screen.getAllByText('Read "docs.northwind.io/billing"').length,
  ).toBeGreaterThan(0);
  expect(screen.queryByText("Waiting for you")).toBeNull();

  rerender(
    readChain(new Map([["call-9", { outcome: "rejected", output: "no" }]])),
  );

  expect(await screen.findByText("Kept out")).toBeDefined();
  expect(
    screen.getAllByText('Read "docs.northwind.io/billing"').length,
  ).toBeGreaterThan(0);
});

test("a released read's row reads Released", async () => {
  render(
    readChain(
      new Map([["call-9", { outcome: "approved", output: { content: "hi" } }]]),
    ),
  );
  expect(await screen.findByText("Released")).toBeDefined();
});

test("a call that may have run claims neither approval nor refusal", async () => {
  render(chain(new Map([["call-7", { outcome: "unknown", output: "" }]])));
  expect(await screen.findByText("Unclear")).toBeDefined();
  expect(screen.queryByText("Approved")).toBeNull();
  expect(screen.queryByText(/Created folder/)).toBeNull();
  expect(screen.queryByText(/Didn't create folder/)).toBeNull();
});

test("a held read whose release is unclear stays a read, not an action that may have run", async () => {
  render(readChain(new Map([["call-9", { outcome: "unknown", output: "" }]])));
  expect(await screen.findByText("Unclear")).toBeDefined();
  expect(
    screen.getAllByText('Read "docs.northwind.io/billing"').length,
  ).toBeGreaterThan(0);
  expect(screen.queryByText(/It may have run/)).toBeNull();
});
