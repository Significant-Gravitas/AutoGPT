import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { CalendarCheckIcon, Task01Icon } from "@hugeicons/core-free-icons";
import { OfficeTemplateExpert } from "../api";

interface Props {
  experts: OfficeTemplateExpert[];
}

export function OfficePreviewList({ experts }: Props) {
  return (
    <ul
      className="flex flex-col divide-y divide-zinc-100"
      aria-label="Experts in this office"
    >
      {experts.map((expert) => (
        <li key={expert.template_id} className="flex items-start gap-3 py-3">
          <ExpertAvatar
            name={expert.name}
            avatarUrl={expert.avatar_url}
            size={36}
          />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Text variant="body-medium" className="text-zinc-900">
                {expert.name}
              </Text>
              <span className="text-xs text-zinc-500">{expert.role}</span>
              {expert.schedule_cron ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-violet-50 px-2 py-0.5 text-[11px] font-medium text-violet-700 ring-1 ring-inset ring-violet-200">
                  <Icon icon={CalendarCheckIcon} size={12} />
                  Scheduled
                </span>
              ) : null}
            </div>
            {expert.tagline ? (
              <Text variant="small" className="mt-0.5 text-zinc-500">
                {expert.tagline}
              </Text>
            ) : null}
            {expert.intro_task_title ? (
              <span className="mt-1.5 flex items-center gap-1.5 text-xs text-zinc-500">
                <Icon
                  icon={Task01Icon}
                  size={13}
                  className="shrink-0 text-zinc-400"
                />
                First task: {expert.intro_task_title}
              </span>
            ) : null}
          </div>
        </li>
      ))}
    </ul>
  );
}
