import { formatElapsedTime } from "../helpers";

type Props = {
  elapsedTime: number;
};

export function RecordingIndicator({ elapsedTime }: Props) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-[3px]">
        {[0, 1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="w-[3px] rounded-full bg-red-500"
            style={{
              animation: `waveform 1s ease-in-out infinite`,
              animationDelay: `${i * 0.1}s`,
              height: "16px",
            }}
          />
        ))}
      </div>
      <span className="min-w-[3ch] text-sm font-medium text-red-500">
        {formatElapsedTime(elapsedTime)}
      </span>
      <style jsx>{`
        @keyframes waveform {
          0%,
          100% {
            transform: scaleY(0.3);
            opacity: 0.5;
          }
          50% {
            transform: scaleY(1);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
