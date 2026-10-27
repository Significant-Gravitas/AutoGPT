"use client";

import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { Button } from "@/components/atoms/Button/Button";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { ExpertAvatar } from "../ExpertAvatar/ExpertAvatar";
import { ACCEPTED_AVATAR_TYPES } from "./helpers";
import { useExpertAvatarPicker } from "./useExpertAvatarPicker";

interface Props {
  name: string;
  category: ExpertAvatarRequestCategory;
  avatarUrl?: string | null;
  onPick: (url: string) => void;
}

export function ExpertAvatarPicker({ name, ...props }: Props) {
  const picker = useExpertAvatarPicker({ name, ...props });
  return (
    <div
      data-slot="expert-avatar-picker"
      className="flex w-full flex-col items-center gap-4 rounded-2xl border border-border bg-background p-5"
    >
      <ExpertAvatar
        name={name || "Your expert"}
        avatarUrl={picker.selectedUrl}
        size={144}
        className={picker.isGenerating ? "animate-pulse" : undefined}
      />
      {picker.isGenerating && (
        <p
          role="status"
          data-slot="expert-avatar-picker-status"
          className="text-body leading-5 text-muted-foreground"
        >
          Creating another avatar. You can keep this one or upload a picture
          while you wait.
        </p>
      )}
      {picker.error && (
        <p
          role="alert"
          data-slot="expert-avatar-picker-error"
          className="text-body leading-5 text-destructive"
        >
          {picker.error}
        </p>
      )}
      <input
        ref={picker.fileInputRef}
        type="file"
        disabled={picker.isUploading}
        accept={ACCEPTED_AVATAR_TYPES}
        aria-label="Upload avatar"
        data-slot="expert-avatar-picker-file-input"
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          void picker.uploadFile(file);
        }}
      />
      <div
        data-slot="expert-avatar-picker-actions"
        className="flex w-full flex-wrap justify-end gap-2"
      >
        <Button
          variant="ghost"
          size="small"
          onClick={picker.openFilePicker}
          disabled={picker.isUploading}
          data-slot="expert-avatar-picker-upload"
        >
          Upload a picture
        </Button>
        <Button
          variant="secondary"
          size="small"
          leadingIcon={RefreshIcon}
          onClick={picker.generate}
          loading={picker.isGenerating}
          disabled={picker.isBusy}
          data-slot="expert-avatar-picker-regenerate"
        >
          Regenerate
        </Button>
        <Button
          size="small"
          onClick={picker.confirm}
          disabled={picker.isUploading}
          data-slot="expert-avatar-picker-confirm"
        >
          Use this avatar
        </Button>
      </div>
    </div>
  );
}
