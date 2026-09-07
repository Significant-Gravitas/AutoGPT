import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { cn } from "@/lib/utils";

interface Props {
  size?: number;
  className?: string;
}

/** AutoPilot's mark. The built-in helper has no avatar image of its own, so
 *  the embossed AutoGPT logo stands in for one wherever it is named. */
export function AutopilotAvatar({ size = 24, className }: Props) {
  const logoSize = Math.round(size * 0.58);

  return (
    <span
      style={{ width: size, height: size }}
      className={cn(
        "flex shrink-0 items-center justify-center rounded-full bg-gradient-to-b from-white to-zinc-100 shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),inset_0_-2px_4px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.06)] ring-1 ring-inset ring-zinc-200/70",
        className,
      )}
    >
      <AutoGPTLogo
        hideText
        viewBox="47 -1 42 42"
        className="h-auto"
        style={{ width: logoSize }}
      />
    </span>
  );
}
