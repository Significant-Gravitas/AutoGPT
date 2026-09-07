import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  type AvatarConfig,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import { SIZE_LADDER, STAGE_SIZE } from "../helpers";
import { StatusToggle } from "./StatusToggle";

interface Props {
  config: AvatarConfig;
  status: AvatarStatus;
  onStatusChange: (status: AvatarStatus) => void;
}

export function AvatarStage({ config, status, onStatusChange }: Props) {
  return (
    <section className="flex flex-col items-center gap-6 rounded-[2rem] bg-zinc-50 px-6 py-10">
      <div
        data-testid="avatar-stage"
        className="flex h-[240px] items-end justify-center"
      >
        <BotAvatar
          config={config}
          status={status}
          size={STAGE_SIZE}
          trackPointer
        />
      </div>
      <StatusToggle status={status} onChange={onStatusChange} />
      <div className="flex items-end gap-5" aria-label="Size ladder">
        {SIZE_LADDER.map((size) => (
          <BotAvatar
            key={size}
            config={config}
            status={status}
            size={size}
            animated={false}
          />
        ))}
      </div>
    </section>
  );
}
