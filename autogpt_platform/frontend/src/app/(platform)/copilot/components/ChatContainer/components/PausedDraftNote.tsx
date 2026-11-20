import { Text } from "@/components/atoms/Text/Text";

export function PausedDraftNote() {
  return (
    <div className="mt-2 px-1">
      <Text tone="muted" variant="small-tight" align="center">
        Your draft is saved. You can keep editing while sending is paused.
      </Text>
    </div>
  );
}
