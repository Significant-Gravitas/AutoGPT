import type { SkippedWebhookPreset } from "@/app/api/__generated__/models/skippedWebhookPreset";
import type { toast } from "@/components/molecules/Toast/use-toast";

// When a new graph version swaps or reconfigures the trigger block, presets
// whose webhook was registered for the old trigger are left pinned to their
// previous version so their existing webhook keeps working. Let the user know
// they need to reconfigure the trigger on the new version.
export function notifySkippedWebhookPresets(
  toastFn: typeof toast,
  skippedPresets: SkippedWebhookPreset[] | undefined | null,
) {
  if (!skippedPresets || skippedPresets.length === 0) return;

  const names = skippedPresets.map((preset) => `"${preset.name}"`).join(", ");
  const single = skippedPresets.length === 1;

  toastFn({
    variant: "info",
    duration: 10000,
    title: single
      ? "A trigger preset needs reconfiguration"
      : "Some trigger presets need reconfiguration",
    description: single
      ? `${names} was kept on its previous version because the new version uses a different trigger. Reconfigure its trigger on the new version to keep it running.`
      : `${names} were kept on their previous version because the new version uses a different trigger. Reconfigure their triggers on the new version to keep them running.`,
  });
}
