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
