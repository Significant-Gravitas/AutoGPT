import { cn } from "@/lib/utils";

/** A faint 40px grid behind the page, fading out toward the edges so the
 *  panels sit on a drafting sheet rather than a flat wash. Colours match
 *  the zinc-50 shell so the fade dissolves into the page. */
export function HomeBackdrop() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 overflow-hidden"
    >
      <div
        className={cn(
          "absolute inset-0 opacity-60",
          "[background-size:40px_40px]",
          "[background-image:linear-gradient(to_right,theme(colors.zinc.200)_1px,transparent_1px),linear-gradient(to_bottom,theme(colors.zinc.200)_1px,transparent_1px)]",
        )}
      />
      <div className="absolute inset-0 bg-zinc-50 [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]" />
    </div>
  );
}
