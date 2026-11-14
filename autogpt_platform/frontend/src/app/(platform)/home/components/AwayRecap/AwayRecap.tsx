"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { Text } from "@/components/atoms/Text/Text";
import { Activity01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useState } from "react";
import { useHomePage } from "../../useHomePage";
import { formatWorkTime } from "../RecentWork/helpers";
import { RecapChart } from "./components/RecapChart";
import { LANES, recapItemsOf, recapWindow, type RecapItem } from "./helpers";

// What happened while the user was away: how much needs them, failed and
// finished, over a small chart of the stretch. Nothing is shown until there
// is something to tell.
export function AwayRecap() {
  const { dashboard } = useHomePage();
  const [activeID, setActiveID] = useState<string | null>(null);
  const [selectedID, setSelectedID] = useState<string | null>(null);

  if (!dashboard) return null;
  const items = recapItemsOf(dashboard);
  if (items.length === 0) return null;

  const { start, end } = recapWindow(dashboard, items);
  const timeZone = dashboard.timezone;
  function timeOf(item: RecapItem) {
    return item.at === null
      ? "Recently"
      : formatWorkTime(new Date(item.at), timeZone);
  }
  // A dot opens its item in place: the title names what it is and the counts
  // give way to the question itself. The same dot, or Back, closes it.
  const selected = items.find((item) => item.id === selectedID) ?? null;
  const selectedLane = selected
    ? LANES.find((lane) => lane.kind === selected.kind)
    : null;
  function toggleSelected(id: string) {
    setSelectedID((current) => (current === id ? null : id));
  }

  return (
    <section
      aria-labelledby="home-away-recap"
      className="mx-auto mb-6 flex w-full max-w-[60rem] flex-col gap-1 rounded-composer-tray bg-surface-tray p-1 text-left"
    >
      <div className="flex h-9 items-center gap-1.5 pr-1 pl-2.5 text-foreground-quiet">
        <Icon
          icon={selectedLane?.icon ?? Activity01Icon}
          size={14}
          tone={selectedLane?.tone ?? "inherit"}
        />
        <Text id="home-away-recap" variant="inbox-medium" as="h2" tone="text">
          {selectedLane?.heading ?? "While you were away"}
        </Text>
        {selected ? (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            withTooltip={false}
            tone="secondary"
            className="ml-auto"
            onClick={() => setSelectedID(null)}
          >
            Back
          </Button>
        ) : null}
      </div>

      <div className="rounded-panel bg-surface shadow-(--inbox-shadow-card)">
        <div className="min-h-[4.25rem] px-4 pt-3" aria-live="polite">
          {selected ? (
            <div
              key={selected.id}
              className="flex flex-col gap-0.5 duration-200 ease-out animate-in fade-in-0 slide-in-from-bottom-1 motion-reduce:animate-none"
            >
              <Text variant="inbox-medium" as="p" tone="ink">
                {selected.title}
              </Text>
              {selected.detail ? (
                <Text
                  variant="inbox"
                  as="p"
                  tone="quiet"
                  className="line-clamp-2"
                >
                  {selected.detail}
                </Text>
              ) : null}
              <div className="mt-1 flex items-center gap-3">
                {selected.href ? (
                  <Link
                    href={selected.href}
                    className="flex items-center gap-0.5 rounded-sm text-body-sm font-medium text-foreground-ink outline-none hover:underline focus-visible:ring-2 focus-visible:ring-focus-inbox"
                  >
                    {selected.actionLabel}
                    <Icon icon={ArrowRight01Icon} size={13} tone="inherit" />
                  </Link>
                ) : null}
                <Text variant="inbox" as="span" tone="faint">
                  {timeOf(selected)}
                </Text>
              </div>
            </div>
          ) : (
            <dl className="flex gap-8">
              {LANES.map((lane) => (
                <div key={lane.kind} className="flex flex-col">
                  <dt className="flex items-center gap-1.5">
                    <span className={`size-1.5 rounded-full ${lane.dot}`} />
                    <Text variant="inbox" as="span" tone="quiet">
                      {lane.label}
                    </Text>
                  </dt>
                  <dd className="text-heading-sm font-medium text-foreground-ink tabular-nums">
                    <NumberTicker
                      value={
                        items.filter((item) => item.kind === lane.kind).length
                      }
                    />
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </div>
        <RecapChart
          items={items}
          start={start}
          end={end}
          startLabel={formatWorkTime(new Date(start), timeZone)}
          timeOf={timeOf}
          activeID={activeID}
          onActive={setActiveID}
          selectedID={selectedID}
          onSelect={toggleSelected}
        />
      </div>
    </section>
  );
}
