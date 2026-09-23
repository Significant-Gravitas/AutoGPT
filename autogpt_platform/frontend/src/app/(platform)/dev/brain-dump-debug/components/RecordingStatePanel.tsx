import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowReloadHorizontalIcon,
  DatabaseIcon,
} from "@hugeicons/core-free-icons";
import {
  formatBytes,
  formatClock,
  formatSeconds,
  formatValue,
  type RecordingSnapshot,
} from "../helpers";
import { DebugField, DebugNote, DebugPanel } from "./DebugPanel";

interface Props {
  snapshot: RecordingSnapshot;
  onRefresh: () => void;
}

export function RecordingStatePanel({ snapshot, onRefresh }: Props) {
  const { meta, parts } = snapshot;

  return (
    <DebugPanel
      title="IndexedDB recording state"
      description="Read live from the autogpt-onboarding-brain-dump database every second."
      icon={DatabaseIcon}
      action={
        <Button
          variant="outline"
          size="small"
          onClick={onRefresh}
          leftIcon={<Icon icon={ArrowReloadHorizontalIcon} size={16} />}
        >
          Refresh
        </Button>
      }
    >
      {!snapshot.supported ? (
        <DebugNote>IndexedDB is not available in this browser.</DebugNote>
      ) : null}
      {snapshot.error ? (
        <DebugNote>Read failed: {snapshot.error}</DebugNote>
      ) : null}

      {meta ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <DebugField label="recordingId" value={meta.recordingId} />
          <DebugField label="mimeType" value={meta.mimeType} />
          <DebugField label="startedAt" value={formatClock(meta.startedAt)} />
          <DebugField
            label="durationSecs"
            value={formatSeconds(meta.durationSecs)}
          />
          <DebugField label="finalized" value={formatValue(meta.finalized)} />
          <DebugField
            label="parts read at"
            value={formatClock(snapshot.readAt)}
          />
        </div>
      ) : (
        <DebugNote>
          No meta row stored — nothing has been recorded in this browser, or the
          dump completed and the store was cleared.
        </DebugNote>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <Badge variant="info">{parts.length} parts</Badge>
        <Badge variant="info">{formatBytes(snapshot.totalBytes)} total</Badge>
      </div>

      {parts.length > 0 ? (
        <div className="mt-4 overflow-hidden rounded-large border border-zinc-200">
          <table className="w-full border-collapse text-left">
            <thead className="bg-zinc-50">
              <tr>
                {["partIndex", "bytes", "savedAt", "uploaded"].map((column) => (
                  <th key={column} className="px-4 py-2">
                    <Text variant="label" className="text-zinc-500">
                      {column}
                    </Text>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parts.map((part) => (
                <tr key={part.partIndex} className="border-t border-zinc-100">
                  <td className="px-4 py-2 font-mono text-sm">
                    {part.partIndex}
                  </td>
                  <td className="px-4 py-2 font-mono text-sm">
                    {formatBytes(part.bytes)}
                  </td>
                  <td className="px-4 py-2 font-mono text-sm">
                    {formatClock(part.savedAt)}
                  </td>
                  <td className="px-4 py-2 font-mono text-sm">
                    {formatValue(part.uploaded)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </DebugPanel>
  );
}
