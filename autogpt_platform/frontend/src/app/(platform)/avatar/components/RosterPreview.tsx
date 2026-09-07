import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  STATUSES,
  type AvatarConfig,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import { rosterConfigs } from "../helpers";
import { SectionHeading } from "./SectionHeading";

interface Props {
  config: AvatarConfig;
  status: AvatarStatus;
}

export function RosterPreview({ config, status }: Props) {
  const roster = [
    { name: "Yours", role: "this expert", status, config },
    ...rosterConfigs(),
  ];
  return (
    <section className="space-y-3">
      <SectionHeading
        title="On the roster"
        hint="how a team reads at list size"
      />
      <ul className="divide-y divide-zinc-100 rounded-2xlarge border border-zinc-200 bg-white">
        {roster.map((member) => {
          const label = STATUSES.find(
            (option) => option.id === member.status,
          )?.label;
          return (
            <li
              key={member.name}
              className="flex items-center gap-3 px-4 py-2.5"
            >
              <BotAvatar
                config={member.config}
                status={member.status}
                size={36}
              />
              <div className="flex min-w-0 flex-1 flex-col">
                <Text
                  variant="small-medium"
                  as="span"
                  className="text-zinc-900"
                >
                  {member.name}
                </Text>
                <Text variant="small" as="span" className="text-zinc-500">
                  {member.role}
                </Text>
              </div>
              <Text variant="small" as="span" className="text-zinc-500">
                {label}
              </Text>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
