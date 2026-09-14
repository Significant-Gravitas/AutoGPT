"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CATEGORY_LABELS,
  PICKABLE_CATEGORIES,
} from "@/components/molecules/NotionAvatar/helpers";
import { NotionAvatar } from "@/components/molecules/NotionAvatar/NotionAvatar";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  DiceFaces01Icon,
} from "@hugeicons/core-free-icons";
import { COLOR_OPTIONS } from "../../../ColorStep/helpers";
import { ACCEPTED_AVATAR_TYPES } from "../../helpers";
import { useNotionAvatarPicker } from "./useNotionAvatarPicker";

interface Props {
  name: string;
  color: string | null;
  onPick: (avatarUrl: string, colorId: string) => void;
}

export function NotionAvatarPicker({ name, color, onPick }: Props) {
  const {
    config,
    colorId,
    setColorId,
    shuffle,
    cycle,
    confirm,
    fileInputRef,
    isUploading,
    openFilePicker,
    handleFileChange,
  } = useNotionAvatarPicker({ name, color, onPick });
  const who = name || "your expert";

  return (
    <div className="ml-auto flex w-full max-w-sm flex-col items-center gap-4 rounded-2xl border border-zinc-200 bg-white p-5">
      <NotionAvatar
        config={config}
        size={132}
        showBadge={false}
        title={`${who}'s face`}
      />

      <div
        role="group"
        aria-label="Expert color"
        className="flex flex-wrap justify-center gap-2"
      >
        {COLOR_OPTIONS.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => setColorId(option.id)}
            aria-label={option.label}
            aria-pressed={option.id === colorId}
            className={cn(
              "size-6 rounded-full transition-transform hover:scale-110",
              option.swatchClassName,
              option.id === colorId &&
                "ring-2 ring-zinc-900 ring-offset-2 ring-offset-white",
            )}
          />
        ))}
      </div>

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
                // The row is labelled inches away, so the tooltip says nothing
                // new — and stacked this tightly it covers the row above.
                withTooltip={false}
                onClick={() => cycle(category, -1)}
              >
                <Icon icon={ArrowLeft01Icon} size={16} />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Next ${CATEGORY_LABELS[category].toLowerCase()}`}
                withTooltip={false}
                onClick={() => cycle(category, 1)}
              >
                <Icon icon={ArrowRight01Icon} size={16} />
              </Button>
            </div>
          </div>
        ))}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_AVATAR_TYPES}
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(event) => handleFileChange(event.target.files?.[0])}
      />

      <div className="flex w-full flex-wrap justify-end gap-2">
        <Button
          variant="ghost"
          size="small"
          className="rounded-full"
          onClick={openFilePicker}
          loading={isUploading}
        >
          Upload a picture
        </Button>
        <Button
          variant="secondary"
          size="small"
          className="rounded-full"
          onClick={shuffle}
          disabled={isUploading}
        >
          <Icon icon={DiceFaces01Icon} size={16} />
          Shuffle
        </Button>
        <Button
          size="small"
          className="rounded-full"
          onClick={confirm}
          disabled={isUploading}
        >
          Use this face
        </Button>
      </div>
    </div>
  );
}
