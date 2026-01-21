import { Text } from "@/components/atoms/Text/Text";

export function ChatLoader() {
  return (
    <Text
      variant="small"
      className="bg-gradient-to-r from-neutral-600 via-neutral-500 to-neutral-600 bg-[length:200%_100%] bg-clip-text text-xs text-transparent [animation:shimmer_2s_ease-in-out_infinite]"
    >
      Taking a bit more time...
    </Text>
  );
}
