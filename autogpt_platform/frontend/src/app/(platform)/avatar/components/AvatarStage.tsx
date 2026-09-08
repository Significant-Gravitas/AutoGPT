import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  type AvatarConfig,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import type { ExpressionId } from "@/components/molecules/BotAvatar/expressions";
import { SIZE_LADDER, STAGE_SIZE, turnToPose, type Turn } from "../helpers";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { StatusToggle } from "./StatusToggle";
import { TurnControl } from "./TurnControl";

interface Props {
  config: AvatarConfig;
  status: AvatarStatus;
  turn: Turn;
  outline: boolean;
  expression: ExpressionId | null;
  onStatusChange: (status: AvatarStatus) => void;
  onTurnChange: (turn: Turn) => void;
  onOutlineChange: (outline: boolean) => void;
}

export function AvatarStage({
  config,
  status,
  turn,
  outline,
  expression,
  onStatusChange,
  onTurnChange,
  onOutlineChange,
}: Props) {
  const poseOffset = turnToPose(turn);
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
          expression={expression ?? undefined}
          trackPointer
          poseOffset={poseOffset}
          outline={outline}
        />
      </div>
      <StatusToggle status={status} onChange={onStatusChange} />
      <TurnControl turn={turn} onChange={onTurnChange} />
      <label className="flex items-center gap-2">
        <Switch
          checked={outline}
          onCheckedChange={onOutlineChange}
          aria-label="Outline"
        />
        <Text variant="small" as="span" className="text-zinc-500">
          Outline
        </Text>
      </label>
      <div className="flex items-end gap-5" aria-label="Size ladder">
        {SIZE_LADDER.map((size) => (
          <BotAvatar
            key={size}
            config={config}
            status={status}
            size={size}
            expression={expression ?? undefined}
            animated={false}
            poseOffset={poseOffset}
            outline={outline}
          />
        ))}
      </div>
    </section>
  );
}
