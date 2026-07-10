"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import {
  CaretDownIcon,
  CaretRightIcon,
  FolderIcon,
  SidebarSimpleIcon,
  TerminalWindowIcon,
  XIcon,
} from "@/components/atoms/AGPTIcon/icons";
import { TerminalConsoleIcon } from "@/components/icons/TerminalConsoleIcon";
import { useState } from "react";
import { useCopilotUIStore } from "../../../../store";
import { ArtifactContent } from "../../../ArtifactPanel/components/ArtifactContent";
import { useArtifactPanel } from "../../../ArtifactPanel/useArtifactPanel";
import { basename } from "../../helpers";
import { DownloadButton } from "../DownloadButton/DownloadButton";
import { FileEditor } from "../FileEditor/FileEditor";
import { TerminalTab } from "../TerminalTab/TerminalTab";
import { FileTree } from "./FileTree";
import { FileTypeIcon } from "./FileTypeIcon";
import { useArtifactTabs } from "./useArtifactTabs";
import { useFilesTab } from "./useFilesTab";

interface Props {
  sessionId: string;
}

export function FilesTab({ sessionId }: Props) {
  const {
    treeData,
    isLoading,
    selectedFilePath,
    openFilePaths,
    selectFile,
    openFile,
    openArtifact,
    closeFile,
    statusByPath,
    loadChildren,
  } = useFilesTab(sessionId);
  const [treeOpen, setTreeOpen] = useState(false);
  const [terminalOpen, setTerminalOpen] = useState(false);
  const closePanel = useCopilotUIStore((s) => s.closeSandboxIdePanel);
  const { activeArtifact, classification, isSourceView, clearArtifactPreview } =
    useArtifactPanel();
  const { openArtifacts, activeArtifactId, selectArtifact, closeArtifact } =
    useArtifactTabs();

  // Opening a sandbox file takes over the editor pane, so drop any artifact
  // preview that was showing there.
  function handleOpenFile(path: string) {
    if (activeArtifact) clearArtifactPreview();
    openFile(path);
  }

  function handleSelectFile(path: string) {
    if (activeArtifact) clearArtifactPreview();
    selectFile(path);
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-3">
        <div className="text-xs text-zinc-400">Waking sandbox…</div>
        <Skeleton className="h-5 w-4/5" />
        <Skeleton className="h-5 w-3/5" />
        <Skeleton className="h-5 w-2/3" />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-1 px-2 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {openArtifacts.map((artifact) => (
            <div
              key={artifact.id}
              onClick={() => selectArtifact(artifact)}
              className={cn(
                "group flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg px-2 py-1 text-sm",
                artifact.id === activeArtifactId
                  ? "bg-zinc-100 text-zinc-900"
                  : "text-zinc-700 hover:bg-zinc-50",
              )}
            >
              <FileTypeIcon name={artifact.title} size={16} />
              <span className="max-w-[10rem] truncate">{artifact.title}</span>
              <button
                type="button"
                aria-label={`Close ${artifact.title}`}
                onClick={(event) => {
                  event.stopPropagation();
                  closeArtifact(artifact);
                }}
                className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-zinc-200 hover:text-zinc-900"
              >
                <XIcon size={12} />
              </button>
            </div>
          ))}
          {openFilePaths.map((path) => {
            return (
              <div
                key={path}
                onClick={() => handleSelectFile(path)}
                className={cn(
                  "group flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg px-2 py-1 text-sm",
                  path === selectedFilePath && !activeArtifact
                    ? "bg-zinc-100 text-zinc-900"
                    : "text-zinc-700 hover:bg-zinc-50",
                )}
              >
                <FileTypeIcon name={basename(path)} size={16} />
                <span className="max-w-[10rem] truncate">{basename(path)}</span>
                <button
                  type="button"
                  aria-label={`Close ${basename(path)}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    closeFile(path);
                  }}
                  className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-zinc-200 hover:text-zinc-900"
                >
                  <XIcon size={12} />
                </button>
              </div>
            );
          })}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <DownloadButton sessionId={sessionId} />
          <button
            type="button"
            aria-label={terminalOpen ? "Hide terminal" : "Show terminal"}
            aria-pressed={terminalOpen}
            onClick={() => setTerminalOpen((value) => !value)}
            className={cn(
              "rounded p-1 text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-zinc-900",
              terminalOpen && "bg-zinc-100 text-zinc-900",
            )}
          >
            <TerminalConsoleIcon size={16} />
          </button>
          <button
            type="button"
            aria-label="Close sandbox panel"
            onClick={closePanel}
            className="rounded p-1 text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
          >
            <SidebarSimpleIcon size={16} />
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2 border-b border-b-zinc-100 px-3 py-2 text-sm text-zinc-700">
        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {activeArtifact ? (
            <span className="truncate text-zinc-700">
              {activeArtifact.title}
            </span>
          ) : selectedFilePath ? (
            selectedFilePath.split("/").map((segment, index) => (
              <span
                key={`${index}-${segment}`}
                className="flex shrink-0 items-center gap-1"
              >
                {index > 0 ? (
                  <CaretRightIcon size={10} className="text-zinc-300" />
                ) : null}
                <span>{segment}</span>
              </span>
            ))
          ) : (
            <span className="text-zinc-400">/</span>
          )}
        </div>
        <button
          type="button"
          aria-label="Toggle file tree"
          aria-pressed={treeOpen}
          onClick={() => setTreeOpen((value) => !value)}
          className={cn(
            "shrink-0 rounded p-1 text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-zinc-900",
            treeOpen && "bg-zinc-100 text-zinc-900",
          )}
        >
          <FolderIcon size={16} />
        </button>
      </div>

      <div className="relative flex min-h-0 flex-1">
        <div className="min-h-0 flex-1 overflow-hidden">
          {activeArtifact && classification ? (
            <ArtifactContent
              artifact={activeArtifact}
              isSourceView={isSourceView}
              classification={classification}
            />
          ) : selectedFilePath ? (
            <FileEditor sessionId={sessionId} path={selectedFilePath} />
          ) : (
            <div className="flex h-full items-center justify-center p-6 text-center text-sm text-zinc-400">
              Open a file from the tree
            </div>
          )}
        </div>
        {/* File tree overlays the editor (does not shift it) and slides in/out. */}
        <div
          className={cn(
            "absolute right-0 top-0 h-full w-60 overflow-hidden border-l border-l-zinc-100 bg-white transition-transform duration-300 ease-out motion-reduce:transition-none",
            treeOpen ? "translate-x-0" : "translate-x-full",
          )}
        >
          <FileTree
            treeData={treeData}
            selectedFilePath={selectedFilePath}
            openFile={handleOpenFile}
            openArtifact={openArtifact}
            statusByPath={statusByPath}
            loadChildren={loadChildren}
          />
        </div>
      </div>

      {terminalOpen ? (
        <div className="flex h-1/4 min-h-0 flex-col border-t border-t-zinc-100">
          <div className="flex items-center justify-between px-3 py-1">
            <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-600">
              <TerminalWindowIcon size={14} />
              Terminal
            </span>
            <button
              type="button"
              aria-label="Hide terminal"
              onClick={() => setTerminalOpen(false)}
              className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-800"
            >
              <CaretDownIcon size={14} />
            </button>
          </div>
          <div className="min-h-0 flex-1">
            <TerminalTab sessionId={sessionId} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
