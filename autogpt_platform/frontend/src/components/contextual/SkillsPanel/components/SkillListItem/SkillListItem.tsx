"use client";

import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  BookOpenIcon,
  DownloadSimpleIcon,
  EyeIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useSkillListItem } from "./useSkillListItem";

interface Props {
  skill: CopilotSkillInfo;
}

export function SkillListItem({ skill }: Props) {
  const {
    descriptionPreview,
    triggers,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
    isDownloading,
    handleDownload,
    isViewOpen,
    openView,
    closeView,
    isDetailLoading,
    detail,
    detailErrorMessage,
  } = useSkillListItem({ skill });

  return (
    <div
      className="flex w-full flex-col gap-3 rounded-large border border-zinc-200 bg-white p-4 sm:flex-row sm:items-start sm:justify-between"
      data-testid="skill-row"
      data-skill-name={skill.name}
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-large border border-slate-50 bg-violet-50">
          <BookOpenIcon size={18} className="text-violet-700" weight="bold" />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="body-medium" className="break-words">
            {skill.name}
          </Text>
          <Text variant="small" className="!text-zinc-500">
            {descriptionPreview}
          </Text>
          {triggers.length > 0 ? (
            <div
              className="mt-1 flex flex-wrap gap-1"
              data-testid="skill-triggers"
            >
              {triggers.map((trigger) => (
                <span
                  key={trigger}
                  className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600"
                >
                  {trigger}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>

      <div className="flex flex-shrink-0 items-center gap-2">
        <Button
          variant="icon"
          size="icon"
          onClick={openView}
          data-testid="skill-view-button"
          aria-label="View skill"
        >
          <EyeIcon className="h-4 w-4" />
        </Button>
        <Button
          variant="icon"
          size="icon"
          onClick={handleDownload}
          loading={isDownloading}
          data-testid="skill-download-button"
          aria-label="Download skill"
        >
          <DownloadSimpleIcon className="h-4 w-4" />
        </Button>
        <Button
          variant="icon"
          size="icon"
          onClick={openDelete}
          data-testid="skill-delete-button"
          aria-label="Delete skill"
        >
          <TrashIcon className="h-4 w-4" />
        </Button>
      </div>

      <Dialog
        controlled={{ isOpen: isViewOpen, set: closeView }}
        styling={{ maxWidth: "48rem" }}
        title={skill.name}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-3">
            <Text variant="small" className="!text-zinc-500">
              {detail?.description ?? descriptionPreview}
            </Text>
            {triggers.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                {triggers.map((trigger) => (
                  <span
                    key={trigger}
                    className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600"
                  >
                    {trigger}
                  </span>
                ))}
              </div>
            ) : null}
            {isDetailLoading ? (
              <div className="flex items-center justify-center py-8">
                <LoadingSpinner />
              </div>
            ) : detailErrorMessage ? (
              <div data-testid="skill-view-error">
                <ErrorCard
                  responseError={{ message: detailErrorMessage }}
                  context={`skill "${skill.name}"`}
                />
              </div>
            ) : (
              <>
                <pre
                  className="max-h-[60vh] overflow-auto rounded-medium bg-zinc-50 p-3 text-sm text-zinc-800"
                  style={{ whiteSpace: "pre-wrap" }}
                  data-testid="skill-view-body"
                >
                  {detail?.body || "(no body)"}
                </pre>
                {detail?.sibling_files && detail.sibling_files.length > 0 ? (
                  <div
                    className="flex flex-col gap-1"
                    data-testid="skill-view-sibling-files"
                  >
                    <Text variant="small" className="!text-zinc-500">
                      Bundled files ({detail.sibling_files.length}
                      ):
                    </Text>
                    <ul className="flex flex-col gap-0.5 pl-3 text-xs text-zinc-600">
                      {detail.sibling_files.map((path) => (
                        <li key={path} className="break-all">
                          {path}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </>
            )}
          </div>
        </Dialog.Content>
      </Dialog>

      <Dialog
        controlled={{ isOpen: isDeleteOpen, set: closeDelete }}
        styling={{ maxWidth: "32rem" }}
        title="Delete skill"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="large">
              Delete the skill <strong>{skill.name}</strong>? Your copilot will
              forget this procedure and can re-distill it later if needed.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeleting}
                onClick={() => closeDelete(false)}
              >
                Keep it
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                loading={isDeleting}
                data-testid="skill-confirm-delete"
              >
                Yes, delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
