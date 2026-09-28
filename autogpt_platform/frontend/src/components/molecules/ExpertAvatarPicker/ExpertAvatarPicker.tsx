"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ACCEPTED_AVATAR_TYPES } from "./helpers";
import { ExpertAvatar } from "../ExpertAvatar/ExpertAvatar";
import { useExpertAvatarPicker } from "./useExpertAvatarPicker";

import { AvatarCatalog } from "./components/AvatarCatalog";
import { GenerationOptions } from "./components/GenerationOptions";

interface Props {
  name: string;
  color: string | null;
  avatarUrl?: string | null;
  categories?: readonly string[] | null;
  onPick: (url: string, color: string) => void;
}

export function ExpertAvatarPicker({ name, ...props }: Props) {
  const picker = useExpertAvatarPicker(props);
  return (
    <div className="flex w-full flex-col items-center gap-4 rounded-2xl border border-border bg-background p-5">
      <ExpertAvatar
        name={name || "Your expert"}
        avatarUrl={picker.selectedUrl}
        size={144}
      />
      <AvatarCatalog
        urls={picker.catalogUrls}
        selectedUrl={picker.selectedUrl}
        disabled={picker.isBusy}
        onSelect={picker.selectCatalog}
      />
      <GenerationOptions
        category={picker.category}
        setCategory={picker.setCategory}
        shape={picker.shape}
        setShape={picker.setShape}
        base={picker.base}
        setBase={picker.setBase}
        tilt={picker.tilt}
        setTilt={picker.setTilt}
        inlay={picker.inlay}
        setInlay={picker.setInlay}
        expression={picker.expression}
        setExpression={picker.setExpression}
        isBusy={picker.isBusy}
      />
      <p className="text-sm text-muted-foreground">
        Keep a managed look, upload a picture, or generate a new clay figure in
        your category&apos;s color. Up to five generations a day, four minutes
        apart.
      </p>
      {picker.isGenerating && (
        <p role="status" className="text-sm text-muted-foreground">
          Creating your avatar. This may take a few minutes.
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
        disabled={picker.isBusy}
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
          disabled={picker.isBusy}
        >
          Upload a picture
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={picker.generate}
          loading={picker.isGenerating}
          disabled={picker.isBusy}
        >
          Generate with AI
        </Button>
        <Button size="small" onClick={picker.confirm} disabled={picker.isBusy}>
          Use this avatar
        </Button>
      </div>
    </div>
  );
}
