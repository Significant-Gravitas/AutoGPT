import type { VoiceSample } from "@/components/organisms/VoicePicker/helpers";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { VoicePicker } from "../VoicePicker";

const samples: VoiceSample[] = [
  { label: "Punchy and bold", text: "Stop guessing what your buyers want." },
  {
    label: "Warm and story-led",
    text: "Every campaign starts with a person, not a product.",
  },
];

function setup(isSubmitting = false) {
  const onPick = vi.fn();
  const onSkip = vi.fn();
  render(
    <VoicePicker
      name="Maria"
      samples={samples}
      onPick={onPick}
      onSkip={onSkip}
      isSubmitting={isSubmitting}
    />,
  );
  return { onPick, onSkip };
}

describe("VoicePicker", () => {
  test("asks how the named expert should write", () => {
    setup();
    expect(screen.getByText("How should Maria write?")).toBeDefined();
    expect(screen.getByText("Punchy and bold")).toBeDefined();
    expect(screen.getByText("Warm and story-led")).toBeDefined();
  });

  test("keeps submit disabled until a choice is made", () => {
    setup();
    const submit = screen.getByRole("button", {
      name: "Use this voice",
    }) as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
  });

  test("picks a preset sample by choice", async () => {
    const { onPick } = setup();
    await userEvent.click(screen.getByText("Punchy and bold"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );
    expect(onPick).toHaveBeenCalledWith({ choice: "a" });
  });

  test("picks the second preset sample by choice", async () => {
    const { onPick } = setup();
    await userEvent.click(screen.getByText("Warm and story-led"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );
    expect(onPick).toHaveBeenCalledWith({ choice: "b" });
  });

  test("picks the user's own pasted sample", async () => {
    const { onPick } = setup();
    await userEvent.type(
      screen.getByPlaceholderText(/Paste a few sentences/),
      "Keep it breezy.",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );
    expect(onPick).toHaveBeenCalledWith({
      choice: "custom",
      customText: "Keep it breezy.",
    });
  });

  test("skips without picking a voice", async () => {
    const { onPick, onSkip } = setup();
    await userEvent.click(screen.getByRole("button", { name: "Skip for now" }));
    expect(onSkip).toHaveBeenCalledOnce();
    expect(onPick).not.toHaveBeenCalled();
  });
});
