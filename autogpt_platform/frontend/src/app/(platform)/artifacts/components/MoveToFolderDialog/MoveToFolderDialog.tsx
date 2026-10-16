"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import { FolderTree } from "./components/FolderTree";
import type { MoveSubject } from "./helpers";
import { useMoveToFolderDialog } from "./useMoveToFolderDialog";

interface Props {
  move: MoveSubject;
  /** What is being moved, as shown in the prompt: `“report.pdf”`, `3 files`
      or `“Reports”`. */
  subject: string;
  /** Offer "Files (root)". For files it defaults to "some file is inside a
      folder", which `currentFolderId` alone cannot express for a mixed
      selection; for a folder, to "not already at the root". */
  canMoveToRoot?: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  onMoved?: () => void;
}

export function MoveToFolderDialog({
  move,
  subject,
  canMoveToRoot,
  isOpen,
  setIsOpen,
  onMoved,
}: Props) {
  const { folders } = useArtifactsFolders();
  const offerRoot = canMoveToRoot ?? defaultCanMoveToRoot(move, folders);
  const {
    rows,
    selectedKey,
    select,
    toggleExpanded,
    confirm,
    canConfirm,
    isMoving,
  } = useMoveToFolderDialog({
    move,
    canMoveToRoot: offerRoot,
    onDone: () => {
      setIsOpen(false);
      onMoved?.();
    },
  });

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "28rem" }}
      title="Move to folder"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-1">
          <Text variant="small" className="mb-1 text-zinc-500">
            Move {subject} to:
          </Text>
          {rows.length === 0 ? (
            <div className="flex h-20 items-center justify-center">
              <Text variant="small" className="text-zinc-400">
                {folders.length === 0 ? "No folders yet" : "No other folders"}
              </Text>
            </div>
          ) : (
            <FolderTree
              rows={rows}
              selectedKey={selectedKey}
              onSelect={select}
              onToggleExpanded={toggleExpanded}
            />
          )}
        </div>
        <Dialog.Footer>
          <Button
            type="button"
            variant="secondary"
            size="small"
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="primary"
            size="small"
            disabled={!canConfirm}
            loading={isMoving}
            onClick={confirm}
            data-testid="confirm-move-to-folder"
          >
            Move
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

function defaultCanMoveToRoot(
  move: MoveSubject,
  folders: { id: string; parent_id?: string | null }[],
): boolean {
  if (move.kind === "files") return move.currentFolderId != null;
  const self = folders.find((folder) => folder.id === move.folderId);
  return (self?.parent_id ?? null) !== null;
}
