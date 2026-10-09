import { fireEvent, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ToolUIPart } from "ai";
import { render } from "@/tests/integrations/test-utils";
import { places, planning } from "@/lib/openui/__tests__/expanded-fixtures";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";

function showResult(source: string, send = vi.fn(), readOnly = false) {
  const part: ToolUIPart = {
    type: "tool-render_ui",
    toolCallId: "expanded-ui",
    state: "output-available",
    input: {},
    output: {
      type: "ui_rendered",
      version: 1,
      session_id: "expanded-session",
      source,
      message: "Chicago visit plan",
    },
  };
  return render(
    <StrictMode>
      <CopilotChatActionsProvider
        onSend={send}
        chatSurface={readOnly ? "share" : "copilot"}
      >
        <ChainMessageParts
          parts={[part]}
          messageID="expanded-message"
          isCurrentlyStreaming={false}
          readOnly={readOnly}
        />
      </CopilotChatActionsProvider>
    </StrictMode>,
  );
}

describe("useful generated views in chat", () => {
  beforeEach(() => sessionStorage.clear());

  it("filters places, selects a real map marker, and discusses it in the current chat", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    showResult(places, send);
    const filter = await screen.findByLabelText("Filter Visit locations");
    fireEvent.change(filter, { target: { value: "Partner" } });
    expect(screen.getByText("1 of 2 places")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "1 River North Customer" }),
    ).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show all" }));
    fireEvent.click(await screen.findByRole("button", { name: "River North" }));
    expect(screen.getByText("Morning customer visit")).toBeDefined();
    expect(screen.getByText("41.89240, -87.63410")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Discuss this place" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain("River North");
    expect(send.mock.calls[0][0]).toContain("41.8924");
    expect(send.mock.calls[0][0]).not.toContain("West Loop");
  });

  it("keeps the place list usable when map tiles fail and restores the selection", async () => {
    const first = showResult(places);
    const marker = await screen.findByRole("button", {
      name: "West Loop",
    });
    const tile = document.querySelector("img.leaflet-tile");
    expect(tile).not.toBeNull();
    fireEvent.error(tile!);
    expect(await screen.findByText(/Map tiles could not load/)).toBeDefined();
    fireEvent.click(marker);
    expect(screen.getByText("Afternoon partner visit")).toBeDefined();
    first.unmount();
    showResult(places);
    expect(await screen.findByText("Afternoon partner visit")).toBeDefined();
  });

  it("restores typed inputs and includes their edited values in a normal follow-up", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    const first = showResult(planning);
    fireEvent.change(await screen.findByLabelText("Visit date"), {
      target: { value: "2026-10-12" },
    });
    fireEvent.change(screen.getByLabelText("Budget"), {
      target: { value: "240" },
    });
    fireEvent.click(screen.getByRole("combobox", { name: "Travel mode" }));
    fireEvent.click(await screen.findByRole("option", { name: "Transit" }));
    expect(
      screen.getByRole("img", { name: /Weekly change.*-2/ }),
    ).toBeDefined();
    expect(
      screen.getByRole("img", { name: /Visit mix.*Customers 3/ }),
    ).toBeDefined();
    expect(screen.getByText("Current")).toBeDefined();
    first.unmount();
    showResult(planning, send);
    expect(
      ((await screen.findByLabelText("Visit date")) as HTMLInputElement).value,
    ).toBe("2026-10-12");
    expect((screen.getByLabelText("Budget") as HTMLInputElement).value).toBe(
      "240",
    );
    expect(
      screen.getByRole("combobox", { name: "Travel mode" }).textContent,
    ).toContain("Transit");
    fireEvent.click(screen.getByRole("button", { name: "Update plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain("2026-10-12");
    expect(send.mock.calls[0][0]).toContain("240");
    expect(send.mock.calls[0][0]).toContain("transit");
  });

  it("allows shared map exploration without sending a conversation action", async () => {
    const send = vi.fn();
    showResult(places, send, true);
    fireEvent.click(
      await screen.findByRole("button", { name: "1 River North Customer" }),
    );
    const action = screen.getByRole("button", { name: "Discuss this place" });
    expect(action.hasAttribute("disabled")).toBe(true);
    fireEvent.click(action);
    expect(send).not.toHaveBeenCalled();
  });
});
