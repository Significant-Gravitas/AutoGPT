"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CATEGORY_LABELS,
  PICKABLE_CATEGORIES,
} from "@/components/molecules/NotionAvatar/helpers";
import { NotionAvatar } from "@/components/molecules/NotionAvatar/NotionAvatar";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  DiceFaces01Icon,
} from "@hugeicons/core-free-icons";
import { useNotionAvatarPicker } from "./useNotionAvatarPicker";

interface Props {
  name: string;
  color: string | null;
  onPick: (avatarUrl: string) => void;
  onCancel: () => void;
}

export function NotionAvatarPicker({ name, color, onPick, onCancel }: Props) {
  const { config, shuffle, cycle, confirm } = useNotionAvatarPicker({
    name,
    color,
    onPick,
  });
  const who = name || "your expert";

  return (
    <div className="ml-auto flex w-full max-w-sm flex-col items-center gap-4 rounded-2xl border border-zinc-200 bg-white p-5">
      <NotionAvatar
        config={config}
        size={132}
        showBadge={false}
        title={`${who}'s face`}
      />

      <div className="flex w-full flex-col gap-1.5">
        {PICKABLE_CATEGORIES.map((category) => (
          <div key={category} className="flex items-center justify-between">
            <Text variant="small" as="span" tone="secondary">
              {CATEGORY_LABELS[category]}
            </Text>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Previous ${CATEGORY_LABELS[category].toLowerCase()}`}
                onClick={() => cycle(category, -1)}
              >
                <Icon icon={ArrowLeft01Icon} size={16} />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Next ${CATEGORY_LABELS[category].toLowerCase()}`}
                onClick={() => cycle(category, 1)}
              >
                <Icon icon={ArrowRight01Icon} size={16} />
              </Button>
            </div>
          </div>
        ))}
      </div>

      <div className="flex w-full flex-wrap justify-end gap-2">
        <Button
          variant="ghost"
          size="small"
          className="rounded-full"
          onClick={onCancel}
        >
          Cancel
        </Button>
        <Button
          variant="secondary"
          size="small"
          className="rounded-full"
          onClick={shuffle}
        >
          <Icon icon={DiceFaces01Icon} size={16} />
          Shuffle
        </Button>
        <Button size="small" className="rounded-full" onClick={confirm}>
          Use this face
        </Button>
      </div>
    </div>
  );
}
