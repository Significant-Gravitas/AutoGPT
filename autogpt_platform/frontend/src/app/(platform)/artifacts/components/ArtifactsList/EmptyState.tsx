import { Text } from "@/components/atoms/Text/Text";
import { FileTypeMarquee } from "./FileTypeMarquee";

interface Props {
  message: string;
}

export function EmptyState({ message }: Props) {
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
