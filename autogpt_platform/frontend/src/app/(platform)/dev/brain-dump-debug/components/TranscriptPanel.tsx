import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import { Text } from "@/components/atoms/Text/Text";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";
import { File02Icon } from "@hugeicons/core-free-icons";
import { TRANSCRIPT_PREVIEW_CHARS } from "../helpers";
import { DebugNote, DebugPanel } from "./DebugPanel";

interface Props {
  finalizeResponse: FinalizeResponse | null;
}

export function TranscriptPanel({ finalizeResponse }: Props) {
  const preview = finalizeResponse?.transcript_preview ?? null;

  return (
    <DebugPanel
      title="Transcript & extraction JSON"
      description="What the pipeline produced from the dump."
      icon={File02Icon}
    >
      <Alert variant="warning">
        <AlertTitle>Not exposed by any current endpoint</AlertTitle>
        <AlertDescription>
          <Text variant="small" className="mb-2 text-zinc-700">
            The full transcript and the extracted CoPilotUnderstanding.data are
            written server-side and never returned to the browser. Verified
            against the whole generated client: the brain-dump tag has only
            parts, finalize, status, recording and discard;{" "}
            <span className="font-mono">GET /onboarding/brain-dump/status</span>{" "}
            returns status, input_mode, error_code and has_audio only, and
            UnderstandingUpdatedResponse is a copilot tool-result shape, not a
            readable endpoint.
          </Text>
          <Text variant="small" className="text-zinc-700">
            Showing them here would need a new backend route — e.g.{" "}
            <span className="font-mono">
              GET /onboarding/brain-dump/transcript
            </span>{" "}
            returning BrainDump.transcript plus transcriptLang, and a read
            endpoint for the user&apos;s CoPilotUnderstanding row. Neither was
            added.
          </Text>
        </AlertDescription>
      </Alert>

      <div className="mt-6">
        <Text variant="label" className="text-zinc-500">
          transcript_preview (from finalize)
        </Text>
        {preview ? (
          <pre className="mt-2 whitespace-pre-wrap rounded-large bg-zinc-50 p-4 font-mono text-sm text-zinc-800">
            {preview}
          </pre>
        ) : (
          <div className="mt-2">
            <DebugNote>
              None captured. Run finalize from the server-status panel above to
              capture one — the server truncates it to{" "}
              {TRANSCRIPT_PREVIEW_CHARS} characters, so even this is not the
              full transcript.
            </DebugNote>
          </div>
        )}
      </div>
    </DebugPanel>
  );
}
