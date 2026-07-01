import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";

import type { CapturedStep } from "../../../hooks/useRecordingWorkflow";
import { RecordingReview } from "../RecordingReview";

function steps(): CapturedStep[] {
  return [
    {
      seq: 1,
      action: "fill",
      label: "First Name",
      activeApp: "Google Chrome",
      value: "John",
    },
    {
      seq: 2,
      action: "fill",
      label: "Email",
      activeApp: "Google Chrome",
      value: "john@x.com",
    },
    { seq: 3, action: "submit", label: "Save", activeApp: "Google Chrome" },
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
    expect(within(dialog).getByText(/First Name/)).toBeDefined();
    expect(within(dialog).getByText("John")).toBeDefined();
    expect(within(dialog).getByText(/Email/)).toBeDefined();
    expect(within(dialog).getByText("john@x.com")).toBeDefined();
  });

  test("makes clear nothing is sent until approval", () => {
    renderReview();
    expect(screen.getByText(/Nothing is sent until you do/i)).toBeDefined();
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
        {
          seq: 1,
          action: "fill",
          label: "SSN",
          activeApp: "Chrome",
          value: null,
          redacted: true,
        },
      ],
    });
    expect(screen.getByText(/value hidden/i)).toBeDefined();
    expect(screen.queryByLabelText(/hide value for step 1/i)).toBeNull();
  });

  test("approve fires the approval gate", () => {
    const { onApprove } = renderReview();
    fireEvent.click(
      screen.getByRole("button", { name: /approve and continue/i }),
    );
    expect(onApprove).toHaveBeenCalledOnce();
  });

  test("approve is disabled when there are no steps", () => {
    renderReview({ steps: [] });
    expect(
      screen.getByRole("button", { name: /approve and continue/i }),
    ).toHaveProperty("disabled", true);
  });
});
