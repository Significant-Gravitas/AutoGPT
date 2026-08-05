import type { DumpStatusResponse } from "@/app/api/__generated__/models/dumpStatusResponse";
import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { AudioWaveformIcon, Download01Icon } from "@hugeicons/core-free-icons";
import {
  formatMs,
  formatValue,
  recordingDownloadHref,
  STATUS_POLL_MS,
} from "../helpers";
import { DebugField, DebugNote, DebugPanel } from "./DebugPanel";

interface Props {
  dump: DumpStatusResponse | null;
  isLoading: boolean;
  isError: boolean;
  finalizeResponse: FinalizeResponse | null;
  finalizeRoundTripMs: number | null;
  finalizeError: string | null;
  canFinalize: boolean;
  isFinalizing: boolean;
  onFinalize: () => void;
}

export function ServerStatusPanel({
  dump,
  isLoading,
  isError,
  finalizeResponse,
  finalizeRoundTripMs,
  finalizeError,
  canFinalize,
  isFinalizing,
  onFinalize,
}: Props) {
  return (
    <DebugPanel
      title="Server status"
      description={`GET /onboarding/brain-dump/status, polled every ${STATUS_POLL_MS}ms.`}
      icon={AudioWaveformIcon}
      action={
        <Link href={recordingDownloadHref()} isExternal variant="secondary">
          <span className="inline-flex items-center gap-2">
            <Icon icon={Download01Icon} size={16} />
            Download server-side recording
          </span>
        </Link>
      }
    >
      {isError ? (
        <DebugNote>
          Status request failed — the endpoint 404s when ONBOARDING_BRAIN_DUMP
          is off for this user, and 401s when signed out.
        </DebugNote>
      ) : null}
      {isLoading && !dump ? <DebugNote>Loading status…</DebugNote> : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <DebugField label="status" value={formatValue(dump?.status)} />
        <DebugField label="input_mode" value={formatValue(dump?.input_mode)} />
        <DebugField label="error_code" value={formatValue(dump?.error_code)} />
        <DebugField label="has_audio" value={formatValue(dump?.has_audio)} />
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Badge variant="info">last finalize response</Badge>
          {finalizeRoundTripMs !== null ? (
            <Badge variant="info">
              round-trip {formatMs(finalizeRoundTripMs)}
            </Badge>
          ) : null}
        </div>
        <Button
          variant="outline"
          size="small"
          onClick={onFinalize}
          disabled={!canFinalize}
          loading={isFinalizing}
        >
          Run finalize on the stored recording
        </Button>
      </div>

      {finalizeError ? (
        <div className="mt-4">
          <DebugNote>Finalize failed: {finalizeError}</DebugNote>
        </div>
      ) : null}

      {finalizeResponse ? (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <DebugField
            label="status"
            value={formatValue(finalizeResponse.status)}
          />
          <DebugField
            label="input_mode"
            value={formatValue(finalizeResponse.input_mode)}
          />
          <DebugField
            label="error_code"
            value={formatValue(finalizeResponse.error_code)}
          />
        </div>
      ) : (
        <div className="mt-4">
          <DebugNote>
            No finalize response captured. Nothing persists the response from
            the real onboarding run, so this page can only show one it issued
            itself. Finalize is idempotent per recording_id and consumes the
            server-side part buffer.
          </DebugNote>
        </div>
      )}
    </DebugPanel>
  );
}
