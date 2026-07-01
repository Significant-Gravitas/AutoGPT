import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { Key, storage } from "@/services/storage/local-storage";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LocalPCRecordingConsent } from "../LocalPCRecordingConsent";
import { hasRememberedRecordingConsent } from "../helpers";

afterEach(() => {
  storage.clean(Key.COPILOT_RECORDING_CLOUD_CONSENT_KINDS);
  vi.clearAllMocks();
});

describe("LocalPCRecordingConsent — calibrated copy (§9.1)", () => {
  function renderDialog(overrides = {}) {
    const props = {
      isOpen: true,
      recordingKind: "screenshots_to_cloud:darwin",
      onSendAndBuild: vi.fn(),
      onKeepLocal: vi.fn(),
      ...overrides,
    };
    render(<LocalPCRecordingConsent {...props} />);
    return props;
  }

  test("renders the calibrated title and the proportionate disclosure", () => {
    renderDialog();
    const dialog = screen.getByRole("dialog");
    expect(
      within(dialog).getByText("Build this skill using the cloud?"),
    ).toBeDefined();
    // States what leaves + why + the deletion.
    expect(
      within(dialog).getByText(/the screen images from this recording go to/i),
    ).toBeDefined();
    expect(
      within(dialog).getByText(
        /a capable model reads them to write the skill/i,
      ),
    ).toBeDefined();
    // Honest comparison to trust already given.
    expect(
      within(dialog).getByText(/the same trust you already place in AutoGPT/i),
    ).toBeDefined();
    // The local alternative, then stop.
    expect(
      within(dialog).getByText(/Install a local model and re-record/i),
    ).toBeDefined();
    // Used to build, not to train.
    expect(within(dialog).getByText(/not to train models/i)).toBeDefined();
  });

  test("uses neither the fear register nor the minimizing register", () => {
    renderDialog();
    const dialog = screen.getByRole("dialog");
    const text = dialog.textContent ?? "";
    // Banned fear register.
    expect(text).not.toMatch(/hacker/i);
    expect(text).not.toMatch(/steal/i);
    expect(text).not.toMatch(/⚠️|🔒/);
    // Banned minimizing register.
    expect(text).not.toMatch(/totally chill/i);
    expect(text).not.toMatch(/nothing to worry about/i);
  });

  test("offers both choices and the remember toggle", () => {
    renderDialog();
    expect(
      screen.getByRole("button", { name: /keep it on my machine/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /send and build/i }),
    ).toBeDefined();
    expect(
      screen.getByLabelText(/remember my choice for recordings like this/i),
    ).toBeDefined();
  });
});

describe("LocalPCRecordingConsent — persistence", () => {
  test("Send without remember does not persist consent", () => {
    const onSendAndBuild = vi.fn();
    render(
      <LocalPCRecordingConsent
        isOpen
        recordingKind="screenshots_to_cloud:darwin"
        onSendAndBuild={onSendAndBuild}
        onKeepLocal={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /send and build/i }));
    expect(onSendAndBuild).toHaveBeenCalledOnce();
    expect(hasRememberedRecordingConsent("screenshots_to_cloud:darwin")).toBe(
      false,
    );
  });

  test("Send WITH remember persists consent per kind", () => {
    const onSendAndBuild = vi.fn();
    render(
      <LocalPCRecordingConsent
        isOpen
        recordingKind="screenshots_to_cloud:darwin"
        onSendAndBuild={onSendAndBuild}
        onKeepLocal={vi.fn()}
      />,
    );
    fireEvent.click(
      screen.getByLabelText(/remember my choice for recordings like this/i),
    );
    fireEvent.click(screen.getByRole("button", { name: /send and build/i }));
    expect(onSendAndBuild).toHaveBeenCalledOnce();
    expect(hasRememberedRecordingConsent("screenshots_to_cloud:darwin")).toBe(
      true,
    );
    // Per-kind, not global: a different kind is still unremembered.
    expect(hasRememberedRecordingConsent("screenshots_to_cloud:windows")).toBe(
      false,
    );
  });

  test("Keep it on my machine declines without persisting", () => {
    const onKeepLocal = vi.fn();
    render(
      <LocalPCRecordingConsent
        isOpen
        recordingKind="screenshots_to_cloud:darwin"
        onSendAndBuild={vi.fn()}
        onKeepLocal={onKeepLocal}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /keep it on my machine/i }),
    );
    expect(onKeepLocal).toHaveBeenCalledOnce();
    expect(hasRememberedRecordingConsent("screenshots_to_cloud:darwin")).toBe(
      false,
    );
  });
});
