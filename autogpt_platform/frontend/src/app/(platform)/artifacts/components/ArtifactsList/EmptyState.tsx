import { Text } from "@/components/atoms/Text/Text";
import { FileTypeMarquee } from "./FileTypeMarquee";

interface Props {
  message: string;
  // When folders are already listed, a full marquee empty state would sit
  // right under them and read as contradictory; show a single quiet line.
  compact?: boolean;
}

export function EmptyState({ message, compact = false }: Props) {
  if (compact) {
    return (
      <Text
        variant="body"
        className="px-2 py-6 text-zinc-500"
        data-testid="artifacts-empty"
      >
        {message}
      </Text>
    );
  }

  return (
    <div
      className="flex min-h-[20rem] flex-col items-center justify-center gap-4 p-8 text-center"
      data-testid="artifacts-empty"
    >
      <FileTypeMarquee />
      <Text variant="h5" className="text-zinc-700">
        {message}
      </Text>
    </div>
  );
}
