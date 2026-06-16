"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { rememberRecordingConsent } from "./helpers";

interface Props {
  /** Whether the dialog is open. The recording flow opens it only for the
   *  `screenshots_to_cloud` route when there's no remembered consent. */
  isOpen: boolean;
  /** Stable per-kind key (see helpers.recordingKind) — what a remembered
   *  "yes" applies to. */
  recordingKind: string;
  /** User chose to send screenshots and build in the cloud. */
  onSendAndBuild: () => void;
  /** User chose to keep everything on their machine (decline). */
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
 * The remembered preference is per-*kind* of recording, not global, and is
 * revocable in settings (see helpers.revokeAllRecordingConsents).
 */
export function LocalPCRecordingConsent({
  isOpen,
  recordingKind,
  onSendAndBuild,
  onKeepLocal,
}: Props) {
  const [remember, setRemember] = useState(false);

  function handleSend() {
    if (remember) {
      rememberRecordingConsent(recordingKind);
    }
    onSendAndBuild();
  }

  return (
    <Dialog
      title="Build this skill using the cloud?"
      styling={{ maxWidth: "34rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          // Dismissing without choosing = keep it local (the safe default).
          if (!open) onKeepLocal();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 py-2">
          <Text variant="body" className="text-neutral-800">
            Your computer doesn&apos;t have a local model that can read these
            screenshots, so building this skill needs AutoGPT&apos;s cloud. If
            you continue, the screen images from this recording go to our
            servers, a capable model reads them to write the skill, then
            they&apos;re deleted (or kept per your data settings).
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
                They&apos;re used to build your skill, not to train models.
              </li>
              <li>
                It&apos;s the same trust you already place in AutoGPT to act on
                your computer, now with screen images for this one recording.
              </li>
            </ul>
          </div>

          <Text variant="body" className="text-sm text-neutral-700">
            Prefer to keep everything on your machine? Install a local model and
            re-record — nothing leaves.
          </Text>

          <label className="flex items-center gap-2 text-sm text-neutral-700">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
              className="h-4 w-4 rounded border-neutral-300"
            />
            Remember my choice for recordings like this
          </label>
        </div>

        <Dialog.Footer className="justify-end">
          <Button variant="secondary" onClick={onKeepLocal}>
            Keep it on my machine
          </Button>
          <Button variant="primary" onClick={handleSend}>
            Send and build
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
