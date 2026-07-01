"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { MagnifyingGlassIcon } from "@phosphor-icons/react";
import { useWorkspaceFilePicker } from "./useWorkspaceFilePicker";
import { WorkspaceFileList } from "./WorkspaceFileList";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (items: WorkspaceFileItem[]) => void;
}

export function WorkspaceFilePicker({ isOpen, onClose, onConfirm }: Props) {
  const picker = useWorkspaceFilePicker({ enabled: isOpen });

  function handleOpenChange(open: boolean) {
    if (!open) {
      picker.reset();
      onClose();
    }
  }

  function handleConfirm() {
    onConfirm(picker.selectedFiles);
    picker.reset();
    onClose();
  }

  const selectedCount = picker.selectedFiles.length;

  return (
    <Dialog
      title="Use a file from your workspace"
      styling={{ maxWidth: "48rem" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <div className="relative flex items-center">
            <MagnifyingGlassIcon
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

          <WorkspaceFileList
            files={picker.files}
            selectedIds={picker.selectedIds}
            onToggle={picker.toggle}
            isLoading={picker.isLoading}
            isError={picker.isError}
            error={picker.error}
            hasMore={picker.hasMore}
            isLoadingMore={picker.isLoadingMore}
            onLoadMore={picker.loadMore}
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
            disabled={selectedCount === 0}
            onClick={handleConfirm}
          >
            {selectedCount > 0 ? `Add ${selectedCount} file(s)` : "Add"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
