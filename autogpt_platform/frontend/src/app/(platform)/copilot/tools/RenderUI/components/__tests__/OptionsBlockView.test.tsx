import type { OptionsBlock } from "@/app/api/__generated__/models/optionsBlock";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { OptionsBlockView } from "../OptionsBlockView";

vi.mock("@/services/analytics/posthog-capture", () => ({
  capturePostHogEvent: vi.fn(),
}));

afterEach(cleanup);

const block: OptionsBlock = {
  type: "options",
  items: [{ title: "Continue" }, { title: "Save for later" }],
};

it("sends only the selected option after an inactive card becomes actionable", async () => {
  const send = Promise.withResolvers<void>();
  const onSend = vi.fn(() => send.promise);

  function card(canAct: boolean) {
    return (
      <CopilotChatActionsProvider onSend={onSend}>
        <OptionsBlockView block={block} canAct={canAct} />
      </CopilotChatActionsProvider>
    );
  }

  const { rerender } = render(card(false));
  const continueButton = screen.getByRole<HTMLButtonElement>("button", {
    name: "Continue",
  });
  const saveButton = screen.getByRole<HTMLButtonElement>("button", {
    name: "Save for later",
  });
  expect(continueButton.disabled).toBe(true);
  expect(saveButton.disabled).toBe(true);
  fireEvent.click(saveButton);
  expect(onSend).not.toHaveBeenCalled();

  rerender(card(true));
  expect(continueButton.disabled).toBe(false);
  expect(saveButton.disabled).toBe(false);
  fireEvent.click(saveButton);
  expect(continueButton.disabled).toBe(true);
  expect(saveButton.disabled).toBe(true);
  fireEvent.click(continueButton);
  expect(onSend).toHaveBeenCalledExactlyOnceWith("Save for later");

  await act(async () => {
    send.resolve();
  });
  expect(continueButton.disabled).toBe(true);
  expect(saveButton.disabled).toBe(true);
});
