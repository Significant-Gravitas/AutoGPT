import { formatElapsedTime } from "../helpers";
import { AudioWaveform } from "./AudioWaveform";

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
        barColor="#ef4444"
        minBarHeight={4}
        maxBarHeight={24}
      />
      <span className="min-w-[3ch] text-sm font-medium text-red-500">
        {formatElapsedTime(elapsedTime)}
      </span>
    </div>
  );
}
