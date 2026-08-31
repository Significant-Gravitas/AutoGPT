import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  CalendarCheckIcon,
  CheckmarkCircle02Icon,
  Clock01Icon,
} from "@hugeicons/core-free-icons";
import { HireOfficeResponse } from "../api";

interface Props {
  result: HireOfficeResponse;
  onDone: () => void;
}

export function HiredRoster({ result, onDone }: Props) {
  const count = result.hired.length;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 rounded-2xl bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
        <Icon icon={CheckmarkCircle02Icon} size={16} />
        {count} expert{count === 1 ? "" : "s"} joined your team and got to work.
      </div>

      <ul
        className="flex flex-col divide-y divide-zinc-100"
        aria-label="Hired experts"
      >
        {result.hired.map((entry) => (
          <li key={entry.expert.id} className="flex items-start gap-3 py-3">
            <ExpertAvatar
              name={entry.expert.name}
              avatarUrl={entry.expert.avatar_url}
              size={36}
            />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <Text variant="body-medium" className="text-zinc-900">
                  {entry.expert.name}
                </Text>
                <span className="text-xs text-zinc-500">
                  {entry.expert.role}
                </span>
              </div>
              {entry.intro_task_title ? (
                <span className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-zinc-600">
                  <Icon
                    icon={Clock01Icon}
                    size={13}
                    className="shrink-0 text-zinc-400"
                  />
                  {entry.intro_task_title}
                  <span className="rounded-full bg-sky-50 px-2 py-0.5 text-[11px] font-medium text-sky-700 ring-1 ring-inset ring-sky-200">
                    {entry.intro_task_id ? "Queued" : "Starting"}
                  </span>
                </span>
              ) : null}
              {entry.schedule_created ? (
                <span className="mt-1 flex items-center gap-1.5 text-xs text-zinc-500">
                  <Icon
                    icon={CalendarCheckIcon}
                    size={13}
                    className="shrink-0 text-zinc-400"
                  />
                  Schedule created
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ul>

      <div className="flex justify-end">
        <Button type="button" variant="primary" onClick={onDone}>
          Done
        </Button>
      </div>
    </div>
  );
}
