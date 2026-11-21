import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { describe, expect, test, vi } from "vitest";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { AIAgentSafetyPopup } from "../AIAgentSafetyPopup";

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

const agent = getGetV2GetLibraryAgentResponseMock({
  id: "agent-1",
  has_human_in_the_loop: true,
  has_sensitive_action: false,
  settings: {
    human_in_the_loop_safe_mode: true,
    sensitive_action_safe_mode: false,
  },
});

describe("AIAgentSafetyPopup", () => {
  test("renders dialog with accessible title when open", () => {
    render(
      <AIAgentSafetyPopup
        agent={agent}
        onAcknowledge={vi.fn()}
        isOpen={true}
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Safety Checks")).toBeDefined();
  });

  test("renders safety content when open", () => {
    render(
      <AIAgentSafetyPopup
        agent={agent}
        onAcknowledge={vi.fn()}
        isOpen={true}
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Safety Checks Enabled")).toBeDefined();
    expect(within(dialog).getByText("Got it")).toBeDefined();
  });

  test("does not render when isOpen is false", () => {
    render(
      <AIAgentSafetyPopup
        agent={agent}
        onAcknowledge={vi.fn()}
        isOpen={false}
      />,
    );

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
