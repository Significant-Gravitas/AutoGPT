import { Text } from "@/components/atoms/Text/Text";

export function PausedDraftNote() {
  return (
    <Text tone="muted" variant="small-tight" className="mt-2 px-1 text-center">
      Your draft is saved. You can keep editing while sending is paused.
    </Text>
  );
}
