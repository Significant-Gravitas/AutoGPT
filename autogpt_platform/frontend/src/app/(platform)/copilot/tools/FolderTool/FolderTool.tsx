"use client";

import type { ToolUIPart } from "ai";
import {
  FileIcon,
  FolderIcon,
  FolderPlusIcon,
  FoldersIcon,
  TrashIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import {
  File as TreeFile,
  Folder as TreeFolder,
  Tree,
  type TreeViewElement,
} from "@/components/molecules/file-tree";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ContentCard,
  ContentCardHeader,
  ContentCardTitle,
  ContentGrid,
  ContentHint,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";
import {
  getAnimationText,
  getFolderToolOutput,
  isAgentsMoved,
  isErrorOutput,
  isFolderCreated,
  isFolderDeleted,
  isFolderList,
  isFolderMoved,
  isFolderUpdated,
  type FolderInfo,
  type FolderToolOutput,
  type FolderTreeInfo,
} from "./helpers";

interface Props {
  part: ToolUIPart;
}

/* ------------------------------------------------------------------ */
/*  Icons                                                              */
/* ------------------------------------------------------------------ */

function ToolStatusIcon({
  isStreaming,
  isError,
}: {
  isStreaming: boolean;
  isError: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <OrbitLoader size={14} />;
  }
  return <FolderIcon size={14} weight="regular" className="text-neutral-400" />;
}

/* ------------------------------------------------------------------ */
/*  Folder card                                                        */
/* ------------------------------------------------------------------ */

function FolderCard({ folder }: { folder: FolderInfo }) {
  return (
    <ContentCard>
      <ContentCardHeader>
        <div className="flex items-center gap-2">
          {folder.color ? (
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ backgroundColor: folder.color }}
            />
          ) : (
            <FolderIcon size={14} weight="fill" className="text-neutral-600" />
          )}
          <ContentCardTitle>{folder.name}</ContentCardTitle>
        </div>
      </ContentCardHeader>
      <ContentHint>
        {folder.agent_count} agent{folder.agent_count !== 1 ? "s" : ""}
        {folder.subfolder_count > 0 &&
          ` · ${folder.subfolder_count} subfolder${folder.subfolder_count !== 1 ? "s" : ""}`}
      </ContentHint>
      {folder.agents && folder.agents.length > 0 && (
        <div className="mt-2 space-y-1 border-t border-neutral-200 pt-2">
          {folder.agents.map((a) => (
            <div key={a.id} className="flex items-center gap-1.5">
              <FileIcon
                size={12}
                weight="duotone"
                className="text-neutral-600"
              />
              <span className="text-xs text-zinc-600">{a.name}</span>
            </div>
          ))}
        </div>
      )}
    </ContentCard>
  );
}

/* ------------------------------------------------------------------ */
/*  Tree renderer using file-tree component                            */
/* ------------------------------------------------------------------ */

type TreeNode = TreeViewElement & { isAgent?: boolean };

function folderTreeToElements(nodes: FolderTreeInfo[]): TreeNode[] {
  return nodes.map((node) => {
    const children: TreeNode[] = [
      ...folderTreeToElements(node.children),
      ...(node.agents ?? []).map((a) => ({
        id: a.id,
        name: a.name,
        isAgent: true,
      })),
    ];
    return {
      id: node.id,
      name: `${node.name} (${node.agent_count} agent${node.agent_count !== 1 ? "s" : ""})`,
      children: children.length > 0 ? children : undefined,
    };
  });
}

function collectAllIDs(nodes: FolderTreeInfo[]): string[] {
  return nodes.flatMap((n) => [n.id, ...collectAllIDs(n.children)]);
}

function FolderTreeView({ tree }: { tree: FolderTreeInfo[] }) {
  const elements = folderTreeToElements(tree);
  const allIDs = collectAllIDs(tree);

  return (
    <Tree
      initialExpandedItems={allIDs}
      elements={elements}
      openIcon={
        <FolderIcon size={16} weight="fill" className="text-neutral-600" />
      }
      closeIcon={
        <FolderIcon size={16} weight="duotone" className="text-neutral-600" />
      }
      className="max-h-64"
    >
      {elements.map((el) => (
        <FolderTreeNodes key={el.id} element={el} />
      ))}
    </Tree>
  );
}

function FolderTreeNodes({ element }: { element: TreeNode }) {
  if (element.isAgent) {
    return (
      <TreeFile
        value={element.id}
        fileIcon={
          <FileIcon size={14} weight="duotone" className="text-neutral-600" />
        }
      >
        <span className="text-sm text-zinc-700">{element.name}</span>
      </TreeFile>
    );
  }

  if (element.children && element.children.length > 0) {
    return (
      <TreeFolder value={element.id} element={element.name} isSelectable>
        {element.children.map((child) => (
          <FolderTreeNodes key={child.id} element={child as TreeNode} />
        ))}
      </TreeFolder>
    );
  }

  return <TreeFolder value={element.id} element={element.name} isSelectable />;
}

/* ------------------------------------------------------------------ */
/*  Accordion content per output type                                  */
/* ------------------------------------------------------------------ */

function AccordionContent({ output }: { output: FolderToolOutput }) {
  if (isFolderCreated(output)) {
    return (
      <ContentGrid>
        <FolderCard folder={output.folder} />
      </ContentGrid>
    );
  }

  if (isFolderList(output)) {
    if (output.tree && output.tree.length > 0) {
      return <FolderTreeView tree={output.tree} />;
    }
    if (output.folders && output.folders.length > 0) {
      return (
        <ContentGrid className="sm:grid-cols-2">
          {output.folders.map((folder) => (
            <FolderCard key={folder.id} folder={folder} />
          ))}
        </ContentGrid>
      );
    }
    return <ContentMessage>No folders found.</ContentMessage>;
  }

  if (isFolderUpdated(output) || isFolderMoved(output)) {
    return (
      <ContentGrid>
        <FolderCard folder={output.folder} />
      </ContentGrid>
    );
  }

  if (isFolderDeleted(output)) {
    return <ContentMessage>{output.message}</ContentMessage>;
  }

  if (isAgentsMoved(output)) {
    return <ContentMessage>{output.message}</ContentMessage>;
  }

  return null;
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

function getAccordionTitle(output: FolderToolOutput): string {
  if (isFolderCreated(output)) return `Created "${output.folder.name}"`;
  if (isFolderList(output))
    return `${output.count} folder${output.count !== 1 ? "s" : ""}`;
  if (isFolderUpdated(output)) return `Updated "${output.folder.name}"`;
  if (isFolderMoved(output)) return `Moved "${output.folder.name}"`;
  if (isFolderDeleted(output)) return "Folder deleted";
  if (isAgentsMoved(output))
    return `Moved ${output.count} agent${output.count !== 1 ? "s" : ""}`;
  return "Folder operation";
}

function getAccordionIcon(output: FolderToolOutput) {
  if (isFolderCreated(output))
    return <FolderPlusIcon size={32} weight="light" />;
  if (isFolderList(output)) return <FoldersIcon size={32} weight="light" />;
  if (isFolderDeleted(output)) return <TrashIcon size={32} weight="light" />;
  return <FolderIcon size={32} weight="light" />;
}

export function FolderTool({ part }: Props) {
  const text = getAnimationText(part);
  const output = getFolderToolOutput(part);

  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const hasContent =
    part.state === "output-available" && !!output && !isErrorOutput(output);

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolStatusIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {hasContent && output && (
        <ToolAccordion
          icon={getAccordionIcon(output)}
          title={getAccordionTitle(output)}
          defaultExpanded={isFolderList(output)}
        >
          <AccordionContent output={output} />
        </ToolAccordion>
      )}
    </div>
  );
}
