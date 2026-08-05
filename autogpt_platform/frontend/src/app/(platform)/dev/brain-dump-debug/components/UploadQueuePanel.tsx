import { Badge } from "@/components/atoms/Badge/Badge";
import { CloudUploadIcon } from "@hugeicons/core-free-icons";
import { formatBytes, type RecordingSnapshot } from "../helpers";
import { DebugField, DebugNote, DebugPanel } from "./DebugPanel";

interface Props {
  snapshot: RecordingSnapshot;
}

export function UploadQueuePanel({ snapshot }: Props) {
  const pending = snapshot.pendingUploads;
  const pendingBytes = snapshot.parts
    .filter((part) => !part.uploaded)
    .reduce((total, part) => total + part.bytes, 0);

  return (
    <DebugPanel
      title="Upload queue"
      description="Derived from the parts table — a part is pending until markPartUploaded flips its flag."
      icon={CloudUploadIcon}
      action={
        <Badge variant={pending === 0 ? "success" : "error"}>
          {pending === 0 ? "drained" : `${pending} pending`}
        </Badge>
      }
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <DebugField label="parts pending upload" value={String(pending)} />
        <DebugField
          label="parts uploaded"
          value={String(snapshot.parts.length - pending)}
        />
        <DebugField label="pending bytes" value={formatBytes(pendingBytes)} />
      </div>
      <div className="mt-4">
        <DebugNote>
          The in-memory queue inside useUploadQueue is not observable from here;
          this panel reflects the durable IndexedDB flags, which is what
          survives a refresh.
        </DebugNote>
      </div>
    </DebugPanel>
  );
}
