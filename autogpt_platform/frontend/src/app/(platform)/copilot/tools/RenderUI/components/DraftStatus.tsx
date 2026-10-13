import { Button } from "@/components/atoms/Button/Button";
import type { useUIDraft } from "../useUIDraft";

interface Props {
  draft: ReturnType<typeof useUIDraft>;
}
export function DraftStatus({ draft }: Props) {
  return (
    <>
      {(!draft.ready || draft.saving || draft.notice) && (
        <p role="status" className="text-xs text-muted-foreground">
          {!draft.ready
            ? "Restoring your draft…"
            : draft.saving
              ? "Saving draft…"
              : draft.notice}
        </p>
      )}
      {draft.syncError && (
        <div role="alert" className="space-y-2 text-sm">
          <p>{draft.syncError}</p>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="small"
              variant="secondary"
              onClick={draft.downloadDraft}
            >
              Download my draft
            </Button>
            {draft.syncConflict ? (
              <Button
                type="button"
                size="small"
                variant="secondary"
                onClick={() => void draft.reloadDraft()}
              >
                Reload saved draft
              </Button>
            ) : (
              <Button
                type="button"
                size="small"
                variant="secondary"
                onClick={() => void draft.retrySync()}
              >
                Retry saving
              </Button>
            )}
          </div>
        </div>
      )}
    </>
  );
}
