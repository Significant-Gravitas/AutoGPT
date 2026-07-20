import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";

import { LocalPCRecordingConsent } from "../LocalPCRecordingConsent";

describe("LocalPCRecordingConsent", () => {
  function renderDialog(overrides = {}) {
    const props = {
      isOpen: true,
      onSendAndBuild: vi.fn(),
      onKeepLocal: vi.fn(),
      ...overrides,
    };
    render(<LocalPCRecordingConsent {...props} />);
    return props;
  }

  test("renders a proportionate deployment-specific disclosure", () => {
    renderDialog();
    const dialog = screen.getByRole("dialog");
    expect(
      within(dialog).getByText("Allow cloud processing for this recording?"),
    ).toBeDefined();
    expect(
      within(dialog).getByText(
        /raw screen images from this recording go to the servers configured/i,
      ),
    ).toBeDefined();
    expect(
      within(dialog).getByText(
        /a capable model reads them to write the skill/i,
      ),
    ).toBeDefined();
    expect(
      within(dialog).getByText(/deployment's data policy defines whether/i),
    ).toBeDefined();
    expect(
      within(dialog).getByText(/the same trust you already place in AutoGPT/i),
    ).toBeDefined();
    expect(
      within(dialog).getByText(/Install a local model and re-record/i),
    ).toBeDefined();
  });

  test("uses neither the fear register nor the minimizing register", () => {
    renderDialog();
    const text = screen.getByRole("dialog").textContent ?? "";
    expect(text).not.toMatch(/hacker/i);
    expect(text).not.toMatch(/steal/i);
    expect(text).not.toMatch(/⚠️|🔒/);
    expect(text).not.toMatch(/totally chill/i);
    expect(text).not.toMatch(/nothing to worry about/i);
  });

  test("requires a fresh choice without offering remembered consent", () => {
    renderDialog();
    expect(
      screen.getByRole("button", { name: /keep screenshots local/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /allow cloud processing/i }),
    ).toBeDefined();
    expect(screen.queryByText(/remember my choice/i)).toBeNull();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  test("submits explicit cloud consent", () => {
    const { onSendAndBuild } = renderDialog();
    fireEvent.click(
      screen.getByRole("button", { name: /allow cloud processing/i }),
    );
    expect(onSendAndBuild).toHaveBeenCalledOnce();
  });

  test("keeps screenshots local when declined", () => {
    const { onKeepLocal } = renderDialog();
    fireEvent.click(
      screen.getByRole("button", { name: /keep screenshots local/i }),
    );
    expect(onKeepLocal).toHaveBeenCalledOnce();
  });
});
