import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { CopilotChatActionsProvider } from "../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainDemo } from "../ChainDemo";

function renderDemo(state: "input-available" | "output-available") {
  return render(
    <CopilotChatActionsProvider onSend={() => {}}>
      <ChainDemo
        part={{
          type: "tool-find_block",
          toolCallId: `call-${state}`,
          state,
          input: { query: "weather" },
          ...(state === "output-available" && {
            output: { blocks: [], count: 0 },
          }),
        }}
      />
    </CopilotChatActionsProvider>,
  );
}

describe("styleguide ChainDemo", () => {
  it("shows a call still waiting for its result open, as the live chain does", () => {
    renderDemo("input-available");
    expect(
      screen.getAllByText('Finding blocks for "weather"…').length,
    ).toBeGreaterThan(0);
    expect(screen.queryAllByRole("button", { expanded: false })).toHaveLength(
      0,
    );
  });

  it("folds a finished call into the chain's summary", () => {
    renderDemo("output-available");
    expect(
      screen.getAllByRole("button", { expanded: false }).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText('Finding blocks for "weather"…')).toBeNull();
    expect(screen.queryAllByRole("button", { expanded: true })).toHaveLength(0);
  });
});
