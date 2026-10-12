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
    <div className="flex w-full flex-col items-center gap-4 rounded-2xl border border-border bg-background p-5">
      <ExpertAvatar
        name={name || "Your expert"}
        avatarUrl={picker.selectedUrl}
        size={144}
        className={picker.isGenerating ? "animate-pulse" : undefined}
      />
      {picker.isGenerating && (
        <p role="status" className="text-sm text-muted-foreground">
          Creating another avatar. You can keep this one or upload a picture
          while you wait.
        </p>
      )}
      {picker.error && (
        <p role="alert" className="text-sm text-destructive">
          {picker.error}
        </p>
      )}
      <input
        ref={picker.fileInputRef}
        type="file"
        disabled={picker.isUploading}
        accept={ACCEPTED_AVATAR_TYPES}
        aria-label="Upload avatar"
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          void picker.uploadFile(file);
        }}
      />
      <div className="flex w-full flex-wrap justify-end gap-2">
        <Button
          variant="ghost"
          size="small"
          onClick={picker.openFilePicker}
          disabled={picker.isUploading}
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
        >
          Regenerate
        </Button>
        <Button
          size="small"
          onClick={picker.confirm}
          disabled={picker.isUploading}
        >
          Use this avatar
        </Button>
      </div>
    </div>
  );
}
