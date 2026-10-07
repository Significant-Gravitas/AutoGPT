import { ExpertDayOneItem } from "@/app/api/__generated__/models/expertDayOneItem";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  items: ExpertDayOneItem[];
  accent: ExpertAccent;
}

export function ExpertDayOne({ name, items, accent }: Props) {
  if (items.length === 0) return null;

  return (
    <ExpertSection title={`What ${name} sets up on day one`}>
      <ol className="flex flex-col divide-y divide-zinc-100 rounded-xl border border-zinc-200 bg-white">
        {items.map((item, index) => (
          <li
            key={`${index}-${item.title}`}
            className="flex items-start gap-3 px-4 py-3.5"
          >
            <span
              aria-hidden="true"
              className={cn(
                "flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold tabular-nums",
                accent.pill,
              )}
            >
              {index + 1}
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-1.5 sm:flex-row sm:items-start sm:gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-base leading-6 font-semibold break-words text-zinc-900">
                  {item.title}
                </div>
                {item.description ? (
                  <Text
                    variant="body"
                    tone="muted"
                    unmask={false}
                    className="leading-6 break-words"
                  >
                    {item.description}
                  </Text>
                ) : null}
              </div>
              {item.timing ? (
                <span className="shrink-0 self-start rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs leading-5 font-medium whitespace-nowrap text-zinc-600">
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
