import { Text } from "@/components/atoms/Text/Text";
import { formatDuration } from "../helpers";

interface Props {
  elapsed: number | null;
  running: boolean;
}

export function RowClock({ elapsed, running }: Props) {
  if (elapsed === null) return null;
  return (
    <Text
      as="span"
      variant="small"
      tone={running ? "text" : "quiet"}
      tabular
      className="whitespace-nowrap"
      title={running ? "Working for" : "Took"}
    >
      {formatDuration(elapsed)}
    </Text>
  );
}
