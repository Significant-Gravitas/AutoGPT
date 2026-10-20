"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { RaiseAttachmentDraft } from "../../helpers";
import { AttachmentAnswer } from "./AttachmentAnswer";
import { KitSearchField } from "./KitSearchField";
import { KitSearchResults } from "./KitSearchResults";
import { SelectedAttachments } from "./SelectedAttachments";
import type { KitSearchScope } from "./helpers";
import { useAttachmentPicker } from "./useAttachmentPicker";

interface Props {
  color: string | null;
  submitted: RaiseAttachmentDraft[] | null;
  isSubmitting?: boolean;
  scope: KitSearchScope;
  existingCount?: number;
  searchLabel: string;
  searchPlaceholder: string;
  emptyQueryHint: string;
  emptyResultsHint: string;
  primaryLabel: string;
  onSubmit: (attachments: RaiseAttachmentDraft[]) => void;
  onSkip: () => void;
}

export function AttachmentStep({
  color,
  submitted,
  isSubmitting = false,
  scope,
  existingCount,
  searchLabel,
  searchPlaceholder,
  emptyQueryHint,
  emptyResultsHint,
  primaryLabel,
  onSubmit,
  onSkip,
}: Props) {
  const picker = useAttachmentPicker({
    scope,
    existingCount,
    onSubmit,
    onSkip,
  });

  if (submitted !== null) {
    return <AttachmentAnswer attachments={submitted} color={color} />;
  }

  return (
    <div className="flex w-full flex-col items-end gap-4">
      <SelectedAttachments
        attachments={picker.attachments}
        color={color}
        onRemove={picker.removeAttachment}
      />
      <KitSearchField
        scope={scope}
        label={searchLabel}
        placeholder={searchPlaceholder}
        value={picker.searchQuery}
        isSearching={picker.isSearching}
        onChange={picker.setSearchQuery}
      />
      <KitSearchResults
        picker={picker}
        emptyQueryHint={emptyQueryHint}
        emptyResultsHint={emptyResultsHint}
      />
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="primary"
          size="small"
          onClick={picker.submit}
          disabled={isSubmitting}
          loading={isSubmitting}
          className="h-[2.625rem] rounded-xl py-3"
        >
          {primaryLabel}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="small"
          onClick={picker.skip}
          disabled={isSubmitting}
          className="h-[2.625rem] rounded-xl py-3"
        >
          Skip
        </Button>
      </div>
    </div>
  );
}
