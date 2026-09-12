"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import type {
  AvatarConfig,
  AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";
import { Icon } from "@/components/atoms/Icon/Icon";
import { DiceIcon, Idea01Icon } from "@hugeicons/core-free-icons";
import { FacetOptionGrid } from "./components/FacetOptionGrid";
import { FacetTabs } from "./components/FacetTabs";
import { useAvatarEditor } from "./useAvatarEditor";

interface Props {
  value: AvatarConfig;
  onChange: (config: AvatarConfig) => void;
  name?: string;
  size?: number;
  status?: AvatarStatus;
  className?: string;
}

export function AvatarEditor({
  value,
  onChange,
  name,
  size = 180,
  status = "idle",
  className,
}: Props) {
  const {
    facet,
    facetId,
    setFacetId,
    active,
    pick,
    move,
    seedName,
    setSeedName,
    surprise,
    seedFromName,
  } = useAvatarEditor({ value, onChange, name });

  return (
    <div
      data-testid="avatar-editor"
      className={cn(
        "flex flex-col gap-6 lg:flex-row lg:items-start lg:gap-8",
        className,
      )}
    >
      <div className="flex shrink-0 flex-col items-center gap-4 lg:w-64">
        <BotAvatar
          config={value}
          size={size}
          status={status}
          trackPointer
          showBadge={status !== "idle"}
          title="Avatar preview"
        />
        <div className="flex w-full flex-col gap-2">
          <Button
            variant="secondary"
            size="small"
            className="w-full rounded-full"
            leftIcon={<Icon icon={DiceIcon} size={16} />}
            onClick={surprise}
          >
            Surprise me
          </Button>
          <Input
            id="avatar-seed-name"
            label="Seed from a name"
            size="small"
            placeholder="Otto"
            value={seedName}
            onChange={(event) => setSeedName(event.target.value)}
          />
          <Button
            variant="secondary"
            size="small"
            className="w-full rounded-full"
            leftIcon={<Icon icon={Idea01Icon} size={16} />}
            disabled={!seedName.trim()}
            onClick={seedFromName}
          >
            Seed from name
          </Button>
        </div>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-4">
        <FacetTabs value={facetId} onChange={setFacetId} />
        <FacetOptionGrid
          facet={facet}
          config={value}
          activeIndex={active}
          onPick={pick}
          onMove={move}
        />
      </div>
    </div>
  );
}
