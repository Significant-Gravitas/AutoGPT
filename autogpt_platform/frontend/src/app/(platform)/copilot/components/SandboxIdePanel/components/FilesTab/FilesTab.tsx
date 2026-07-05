"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { CaretRightIcon, FileIcon, FolderIcon } from "@phosphor-icons/react";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { Tree, type NodeRendererProps } from "react-arborist";
import { FileEditor } from "../FileEditor/FileEditor";
import { type SandboxTreeNode, useFilesTab } from "./useFilesTab";

interface FilesTabContextValue {
  selectedFilePath: string | null;
  selectFile: (path: string | null) => void;
  statusByPath: Map<string, string>;
  loadChildren: (path: string) => void;
}

const FilesTabContext = createContext<FilesTabContextValue | null>(null);

function useFilesTabContext() {
  const ctx = useContext(FilesTabContext);
  if (!ctx) throw new Error("FileTreeRow must be used within FilesTab");
  return ctx;
}

function FileTreeRow({ node, style }: NodeRendererProps<SandboxTreeNode>) {
  const { selectedFilePath, selectFile, statusByPath, loadChildren } =
    useFilesTabContext();
  const status = statusByPath.get(node.data.id);

  function handleClick() {
    if (node.data.isDir) {
      if (!node.isOpen) loadChildren(node.data.id);
      node.toggle();
    } else {
      selectFile(node.data.id);
    }
  }

  return (
    <div
      style={style}
      onClick={handleClick}
      className="flex h-full cursor-pointer items-center gap-1 rounded px-1 text-sm text-zinc-700 hover:bg-zinc-100"
    >
      {node.data.isDir ? (
        <>
          <CaretRightIcon
            size={12}
            className={cn(
              "shrink-0 text-zinc-400 transition-transform",
              node.isOpen && "rotate-90",
            )}
          />
          <FolderIcon size={14} className="shrink-0 text-zinc-500" />
        </>
      ) : (
        <>
          <span className="w-3 shrink-0" />
          <FileIcon size={14} className="shrink-0 text-zinc-400" />
        </>
      )}
      <span
        className={cn(
          "truncate",
          selectedFilePath === node.data.id && "font-medium text-purple-600",
        )}
      >
        {node.data.name}
      </span>
      {status ? (
        <span className="ml-auto shrink-0 pr-1 font-mono text-[0.6875rem] font-semibold text-zinc-400">
          {status}
        </span>
      ) : null}
    </div>
  );
}

interface Props {
  sessionId: string;
}

export function FilesTab({ sessionId }: Props) {
  const {
    treeData,
    isLoading,
    selectedFilePath,
    selectFile,
    statusByPath,
    loadChildren,
  } = useFilesTab(sessionId);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      setSize({ width: element.clientWidth, height: element.clientHeight });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

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
    <FilesTabContext.Provider
      value={{ selectedFilePath, selectFile, statusByPath, loadChildren }}
    >
      <div className="flex h-full min-h-0 flex-col">
        <div
          ref={containerRef}
          className={cn(
            "min-h-0 overflow-hidden",
            selectedFilePath ? "h-2/5" : "flex-1",
          )}
        >
          <Tree
            data={treeData}
            width={size.width || 320}
            height={size.height || 400}
            rowHeight={28}
            indent={12}
            openByDefault={false}
            disableDrag
            disableDrop
            disableEdit
          >
            {FileTreeRow}
          </Tree>
        </div>
        {selectedFilePath ? (
          <div className="min-h-0 flex-1 border-t border-t-zinc-100">
            <FileEditor sessionId={sessionId} path={selectedFilePath} />
          </div>
        ) : null}
      </div>
    </FilesTabContext.Provider>
  );
}
