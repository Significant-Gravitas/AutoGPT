import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  tagline?: string | null;
  compact?: boolean;
}

export function ExpertTagline({ tagline, compact }: Props) {
  if (!tagline?.trim()) return null;

  return (
    <Text
      variant="body"
      tone="secondary"
      unmask={false}
      className={cn("mt-2", compact && "line-clamp-2 min-h-[2lh]")}
    >
      {tagline}
    </Text>
  );
}
