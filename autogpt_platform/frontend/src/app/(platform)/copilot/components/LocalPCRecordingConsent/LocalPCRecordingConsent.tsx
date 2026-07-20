"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  isSubmitting?: boolean;
  errorMessage?: string | null;
  onSendAndBuild: () => void;
  onKeepLocal: () => void;
}

/**
 * The calibrated cloud-fallback consent dialog
 * (WORKFLOW_RECORDING.md §9.1).
 *
 * This appears ONLY for the `screenshots_to_cloud` interpretation route —
 * the one case where raw screen images would leave the machine. The
 * default `extract_then_cloud` (text/structure, no pixels) and the local
 * routes do not prompt (§3.1).
 *
 * The copy is a spec requirement, not a UI detail: the tone is the
 * control. It uses the *calibrated register* — state what leaves, why it
 * helps, the realistic scope, an honest comparison to trust already given,
 * and the local alternative, then stop. Deliberately:
 *   - NO warning iconography (⚠️/🔒) — it biases toward the fear register
 *     before the user has read a word.
 *   - No fear register ("hackers could steal your data").
 *   - No minimizing register ("totally chill thing to hit yes on").
 *
 * v1 never remembers this decision. Every screenshot transfer requires a new
 * prompt for the recording being reviewed.
 */
export function LocalPCRecordingConsent({
  isOpen,
  isSubmitting = false,
  errorMessage,
  onSendAndBuild,
  onKeepLocal,
}: Props) {
  return (
    <Dialog
      title="Allow cloud processing for this recording?"
      styling={{ maxWidth: "34rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onKeepLocal();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 py-2">
          <Text variant="body" className="text-neutral-800">
            Your computer doesn&apos;t have a local model that can read these
            screenshots, so later skill generation needs cloud processing
            through this AutoGPT deployment. If you ask Copilot to generate the
            skill, the raw screen images from this recording go to the servers
            configured for this deployment, a capable model reads them to write
            the skill, and your deployment&apos;s data policy and configured
            data settings govern how they are retained and used.
          </Text>

          <div className="flex flex-col gap-1">
            <Text variant="body" className="font-medium text-neutral-900">
              Worth knowing:
            </Text>
            <ul className="ml-4 list-disc text-sm text-neutral-700">
              <li>
                The images show whatever was on your screen while recording —
                including anything else that was open. You decide what&apos;s
                visible.
              </li>
              <li>
                Your deployment&apos;s data policy defines whether submitted
                data may be used for model improvement.
              </li>
              <li>
                It&apos;s the same trust you already place in AutoGPT to act on
                your computer, now with screen images for this one recording.
              </li>
            </ul>
          </div>

          <Text variant="body" className="text-sm text-neutral-700">
            Prefer to keep raw screenshots on your machine? Install a local
            model and re-record. The hygiene-redacted structured steps shown in
            review have already been transferred to this authenticated browser.
          </Text>

          {errorMessage ? (
            <Text variant="body" role="alert" className="text-sm text-red-700">
              {errorMessage}
            </Text>
          ) : null}
        </div>

        <Dialog.Footer className="flex-wrap justify-end">
          <Button
            variant="secondary"
            disabled={isSubmitting}
            onClick={onKeepLocal}
          >
            Keep screenshots local
          </Button>
          <Button
            variant="primary"
            disabled={isSubmitting}
            loading={isSubmitting}
            onClick={onSendAndBuild}
          >
            Allow cloud processing
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
