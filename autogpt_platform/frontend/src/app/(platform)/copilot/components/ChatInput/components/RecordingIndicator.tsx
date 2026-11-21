import { formatElapsedTime } from "../helpers";
import { AudioWaveform } from "./AudioWaveform";
import { Text } from "@/components/atoms/Text/Text";

type Props = {
  elapsedTime: number;
  audioStream: MediaStream | null;
};

export function RecordingIndicator({ elapsedTime, audioStream }: Props) {
  return (
    <div className="flex items-center gap-3">
      <AudioWaveform
        stream={audioStream}
        barCount={20}
        barWidth={3}
        barGap={2}
        barColor="rgb(var(--color-error))"
        minBarHeight={4}
        maxBarHeight={24}
      />
      <Text
        variant="body-medium-tight"
        as="span"
        tone="error"
        className="min-w-[3ch]"
      >
        {formatElapsedTime(elapsedTime)}
      </Text>
    </div>
  );
}
