import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";

import type { CapturedStep } from "../../../hooks/recording-helpers";
import { RecordingReview } from "../RecordingReview";

function step(overrides: Partial<CapturedStep>): CapturedStep {
  return {
    seq: 1,
    timestamp: 0,
    actor: "human",
    action: "click",
    enrichment: { kind: "none", selectors: [] },
    outcome: "unknown",
    ...overrides,
  };
}

function steps(): CapturedStep[] {
  return [
    step({
      seq: 1,
      timestamp: 123.5,
      action: "fill",
      label: "First Name",
      activeApp: "Google Chrome",
      activeWindow: "Customer form",
      screenshotRef: "frame-1",
      cursor: [120, 240],
      narration: "Enter the customer's first name",
      enrichment: {
        kind: "dom",
        selectors: [{ strategy: "css", value: "#first-name" }],
        axPath: "Application/Window/TextField",
        role: "textbox",
        label: "First Name",
        url: "https://example.test/customers/new",
      },
      value: "John",
      valueType: "text",
      isParameter: true,
      outcome: "ok",
    }),
    step({
      seq: 2,
      action: "fill",
      label: "Email",
      activeApp: "Google Chrome",
      value: "john@x.com",
    }),
    step({
      seq: 3,
      action: "submit",
      label: "Save",
      activeApp: "Google Chrome",
    }),
  ];
}

function renderReview(overrides = {}) {
  const props = {
    isOpen: true,
    steps: steps(),
    onDeleteStep: vi.fn(),
    onRedactStep: vi.fn(),
    onApprove: vi.fn(),
    onCancel: vi.fn(),
    ...overrides,
  };
  render(<RecordingReview {...props} />);
  return props;
}

describe("RecordingReview", () => {
  test("lists the captured steps with their values", () => {
    renderReview();
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getAllByText(/First Name/).length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText("John").length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText(/Email/).length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText("john@x.com").length).toBeGreaterThan(0);
  });

  test("explains the transferred review data and raw screenshot boundary", () => {
    renderReview();
    expect(
      screen.getByText(/transferred these hygiene-redacted structured steps/i),
    ).toBeDefined();
    expect(
      screen.getByText(/raw screenshots stay on your machine unless/i),
    ).toBeDefined();
  });

  test("exposes all retained trajectory metadata in accessible details", () => {
    renderReview();
    const row = screen.getByTestId("recording-step-1");
    fireEvent.click(
      within(row).getByLabelText(/view retained metadata for step 1/i),
    );

    expect(within(row).getByText("frame-1")).toBeDefined();
    expect(within(row).getByText("x=120, y=240")).toBeDefined();
    expect(within(row).getByText("Customer form")).toBeDefined();
    expect(
      within(row).getByText("Enter the customer's first name"),
    ).toBeDefined();
    expect(within(row).getByText("Application/Window/TextField")).toBeDefined();
    expect(within(row).getByText("textbox")).toBeDefined();
    expect(
      within(row).getByText("https://example.test/customers/new"),
    ).toBeDefined();
    expect(within(row).getByText("css: #first-name")).toBeDefined();
    expect(within(row).getByText("Redacted")).toBeDefined();
    expect(within(row).getByText("No")).toBeDefined();
  });

  test("delete removes a step via the callback", () => {
    const { onDeleteStep } = renderReview();
    fireEvent.click(screen.getByLabelText(/delete step 1/i));
    expect(onDeleteStep).toHaveBeenCalledWith(1);
  });

  test("redact hides a step value via the callback", () => {
    const { onRedactStep } = renderReview();
    fireEvent.click(screen.getByLabelText(/hide value for step 2/i));
    expect(onRedactStep).toHaveBeenCalledWith(2);
  });

  test("a redacted step shows 'value hidden' and no redact button", () => {
    renderReview({
      steps: [
        step({
          seq: 1,
          action: "fill",
          label: "SSN",
          activeApp: "Chrome",
          value: null,
          redacted: true,
        }),
      ],
    });
    expect(screen.getByText(/value hidden/i)).toBeDefined();
    expect(screen.queryByLabelText(/hide value for step 1/i)).toBeNull();
  });

  test("approve fires the approval gate", () => {
    const { onApprove } = renderReview();
    fireEvent.click(screen.getByRole("button", { name: /finish review/i }));
    expect(onApprove).toHaveBeenCalledOnce();
  });

  test("approve is disabled when there are no steps", () => {
    renderReview({ steps: [] });
    expect(
      screen.getByRole("button", { name: /finish review/i }),
    ).toHaveProperty("disabled", true);
  });

  test("disables review actions while the authoritative review is applied", () => {
    renderReview({ isSubmitting: true });
    expect(
      screen.getByRole("button", { name: /finish review/i }),
    ).toHaveProperty("disabled", true);
    expect(screen.getByRole("button", { name: /cancel/i })).toHaveProperty(
      "disabled",
      true,
    );
  });
});
