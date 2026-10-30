import { ExpertDayOneItem } from "@/app/api/__generated__/models/expertDayOneItem";
import { cn } from "@/lib/utils";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  items: ExpertDayOneItem[];
  accent: ExpertAccent;
}

/** Day one as a timeline: numbered steps joined by a line, each with its
 *  timing on the right. */
export function ExpertDayOne({ name, items, accent }: Props) {
  if (items.length === 0) return null;

  return (
    <ExpertSection title={`What ${name} sets up on day one`}>
      <ol className="rounded-2xl bg-surface p-5 smooth-shadow-ring-sm">
        {items.map((item, index) => (
          <li
            key={`${index}-${item.title}`}
            className="relative flex items-start gap-4 pb-6 last:pb-0"
          >
            {index < items.length - 1 ? (
              <span
                aria-hidden
                className="absolute top-9 left-[15px] h-[calc(100%-2.25rem)] w-px bg-surface-active"
              />
            ) : null}
            <span
              aria-hidden="true"
              className={cn(
                "relative flex size-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold tabular-nums ring-4 ring-surface",
                accent.pill,
              )}
            >
              {index + 1}
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-1.5 pt-1 sm:flex-row sm:items-start sm:gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-base leading-6 font-semibold break-words text-foreground-strong">
                  {item.title}
                </div>
                {item.description ? (
                  <p className="mt-0.5 text-body-md leading-6 break-words text-foreground-tertiary">
                    {item.description}
                  </p>
                ) : null}
              </div>
              {item.timing ? (
                <span className="shrink-0 self-start rounded-full border border-border-base bg-gradient-to-b from-surface to-surface-muted px-2.5 py-0.5 text-xs leading-5 font-medium whitespace-nowrap text-foreground-secondary">
                  {item.timing}
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </ExpertSection>
  );
}
