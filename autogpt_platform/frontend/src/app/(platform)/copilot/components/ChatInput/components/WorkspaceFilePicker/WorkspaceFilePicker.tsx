"use client";

import { FolderBreadcrumb } from "@/app/(platform)/artifacts/components/WorkspaceFolders/FolderBreadcrumb";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  useWorkspaceFilePicker,
  type PickedItem,
} from "./useWorkspaceFilePicker";
import { WorkspaceFileList } from "./WorkspaceFileList";
import { WorkspaceFolderRows } from "./WorkspaceFolderRows";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (items: PickedItem[]) => void;
  expertId?: string | null;
  /** Names the expert in the filter row; falls back to "this expert". */
  expertName?: string | null;
}

export function WorkspaceFilePicker({
  isOpen,
  onClose,
  onConfirm,
  expertId,
  expertName,
}: Props) {
  const picker = useWorkspaceFilePicker({ enabled: isOpen, expertId });

  function handleOpenChange(open: boolean) {
    if (!open) {
      picker.reset();
      onClose();
    }
  }

  function handleConfirm() {
    onConfirm(picker.selectedItems);
    picker.reset();
    onClose();
  }

  const who = expertName ?? "this expert";
  const { selectedFileCount: files, selectedFolderCount: folders } = picker;

  return (
    <Dialog
      title="Use a file from your workspace"
      styling={{ maxWidth: "48rem" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <div className="relative flex items-center">
            <Icon
              icon={Search01Icon}
              width={18}
              height={18}
              className="absolute left-4 top-1/2 z-20 -translate-y-1/2 text-zinc-500"
            />
            <Input
              label="Search workspace files"
              id="workspace-file-picker-search"
              hideLabel
              type="text"
              value={picker.searchTerm}
              onChange={(e) => picker.setSearchTerm(e.target.value)}
              placeholder="Search files"
              className="w-full pl-12"
              wrapperClassName="!mb-0"
            />
          </div>

          {expertId ? (
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Switch
                  id="picker-expert-only"
                  checked={picker.expertOnly}
                  onCheckedChange={picker.setExpertOnly}
                />
                <label htmlFor="picker-expert-only">
                  <Text
                    variant="small-medium"
                    as="span"
                    className="text-zinc-700"
                  >
                    Only {who}&rsquo;s files
                  </Text>
                </label>
              </div>
              <Text variant="small" className="text-zinc-500">
                {picker.expertOnly
                  ? `From your chats with ${who}`
                  : "Plus your own files and folders"}
              </Text>
            </div>
          ) : null}

          {picker.showFolders && picker.breadcrumb.length > 0 ? (
            <FolderBreadcrumb
              items={picker.breadcrumb}
              onNavigate={picker.openFolder}
              compact
            />
          ) : null}

          {picker.showFolders ? (
            <WorkspaceFolderRows
              folders={picker.folderRows}
              allFolders={picker.folders}
              selectedIds={picker.selectedFolderIds}
              isLoading={picker.isFoldersLoading}
              onOpen={picker.openFolder}
              onToggleAttach={picker.toggleFolder}
            />
          ) : null}

          <WorkspaceFileList
            files={picker.files}
            selectedIds={picker.selectedIds}
            onSelect={picker.select}
            isLoading={picker.isLoading}
            isError={picker.isError}
            error={picker.error}
            hasMore={picker.hasMore}
            isLoadingMore={picker.isLoadingMore}
            onLoadMore={picker.loadMore}
            emptyMessage={emptyMessage({
              isExpertOnly: !!expertId && picker.expertOnly,
              who,
              isSearching: picker.searchTerm.trim().length > 0,
              isInFolder: picker.folderId !== null,
              hasFolderRows: picker.folderRows.length > 0,
            })}
            emptyAction={
              expertId && picker.expertOnly ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="small"
                  onClick={() => picker.setExpertOnly(false)}
                >
                  Show your own files too
                </Button>
              ) : null
            }
          />
        </div>

        <Dialog.Footer>
          <Button
            type="button"
            variant="secondary"
            size="small"
            onClick={() => handleOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="primary"
            size="small"
            disabled={files + folders === 0}
            onClick={handleConfirm}
          >
            {addLabel(files, folders)}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

function addLabel(files: number, folders: number): string {
  const parts: string[] = [];
  if (files > 0) parts.push(`${files} ${files === 1 ? "file" : "files"}`);
  if (folders > 0)
    parts.push(`${folders} ${folders === 1 ? "folder" : "folders"}`);
  return parts.length > 0 ? `Add ${parts.join(" and ")}` : "Add";
}

function emptyMessage(opts: {
  isExpertOnly: boolean;
  who: string;
  isSearching: boolean;
  isInFolder: boolean;
  hasFolderRows: boolean;
}): string {
  if (opts.isExpertOnly)
    return opts.isSearching
      ? `No matching files from ${opts.who}.`
      : `No files from ${opts.who} yet.`;
  if (opts.isSearching) return "No matching files.";
  if (opts.isInFolder)
    return opts.hasFolderRows
      ? "No files directly in this folder."
      : "This folder is empty.";
  return "No files in your workspace yet.";
}
