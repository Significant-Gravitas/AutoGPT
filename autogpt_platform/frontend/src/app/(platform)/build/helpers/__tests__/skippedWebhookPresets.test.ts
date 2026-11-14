import { describe, expect, it, vi } from "vitest";
import type { toast } from "@/components/molecules/Toast/use-toast";
import { notifySkippedWebhookPresets } from "../skippedWebhookPresets";

// This helper backs both the builder-save path (useSaveGraph) and the version
// revert path (useBuilderChatPanel), so covering it directly exercises the
// notification logic for every activation flow that surfaces skipped presets.
describe("notifySkippedWebhookPresets", () => {
  it("does not toast when there are no skipped presets", () => {
    const toastFn = vi.fn() as unknown as typeof toast;
    notifySkippedWebhookPresets(toastFn, []);
    notifySkippedWebhookPresets(toastFn, undefined);
    notifySkippedWebhookPresets(toastFn, null);
    expect(toastFn).not.toHaveBeenCalled();
  });

  it("warns with a singular message for one skipped preset", () => {
    const toastFn = vi.fn() as unknown as typeof toast;
    notifySkippedWebhookPresets(toastFn, [
      { id: "p1", name: "GitHub PR trigger", pinned_version: 1 },
    ]);
    expect(toastFn).toHaveBeenCalledTimes(1);
    expect(toastFn).toHaveBeenCalledWith(
      expect.objectContaining({
        variant: "info",
        title: expect.stringContaining("A trigger preset"),
        description: expect.stringContaining("GitHub PR trigger"),
      }),
    );
  });

  it("warns with a plural message listing all skipped presets", () => {
    const toastFn = vi.fn() as unknown as typeof toast;
    notifySkippedWebhookPresets(toastFn, [
      { id: "p1", name: "Telegram trigger", pinned_version: 2 },
      { id: "p2", name: "GitHub trigger", pinned_version: 2 },
    ]);
    const arg = (toastFn as unknown as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(arg.title).toContain("Some trigger presets");
    expect(arg.description).toContain("Telegram trigger");
    expect(arg.description).toContain("GitHub trigger");
  });
});
