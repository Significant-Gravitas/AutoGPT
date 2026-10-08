import Link from "next/link";
import { cn } from "@/lib/utils";

interface Props {
  mode: "sample" | "live";
  onMode: (mode: "sample" | "live") => void;
  liveAvailable: boolean;
  standalone: boolean;
}

export function ModeSelector({
  mode,
  onMode,
  liveAvailable,
  standalone,
}: Props) {
  return (
    <div className="border-b border-zinc-100 px-5 py-3">
      <div
        className="inline-flex gap-1 rounded-lg bg-zinc-100 p-1"
        aria-label="Generation mode"
      >
        {(["sample", "live"] as const).map((value) => (
          <button
            key={value}
            aria-pressed={mode === value}
            onClick={() => onMode(value)}
            className={cn(
              "rounded-md px-3 py-1 text-[11px] font-medium text-zinc-500 transition-colors focus-visible:outline-purple-500",
              mode === value && "bg-white text-zinc-800 shadow-sm",
            )}
          >
            {value === "sample" ? "Sample" : "Live AI"}
          </button>
        ))}
      </div>
      {mode === "live" && !liveAvailable && (
        <p className="mt-3 text-xs leading-relaxed text-zinc-500">
          {standalone ? (
            <>
              Live generation is available in your signed-in workspace.{" "}
              <Link
                href="/copilot/openui"
                className="font-medium text-purple-600 underline underline-offset-2"
              >
                Open the lab
              </Link>
            </>
          ) : (
            "Live AI isn't configured in this environment yet. You can explore all three workflows in Sample mode."
          )}
        </p>
      )}
    </div>
  );
}
