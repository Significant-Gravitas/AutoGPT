import { Text } from "@/components/atoms/Text/Text";

export function PausedDraftNote() {
  return (
    <Text
      variant="small"
      className="mt-2 px-1 text-center !text-xs !text-zinc-500"
    >
      Your draft is saved. You can keep editing while sending is paused.
    </Text>
  );
}
