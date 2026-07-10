"use client";

import { cn } from "@/lib/utils";
import {
  CaretRightIcon,
  MagnifyingGlassIcon,
} from "@/components/atoms/AGPTIcon/icons";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { Tree, type NodeRendererProps } from "react-arborist";
import type { ArtifactRef } from "../../../../store";
import { FileTypeIcon } from "./FileTypeIcon";
import { type SandboxTreeNode } from "./useFilesTab";

interface FileTreeContextValue {
  selectedFilePath: string | null;
  openFile: (path: string) => void;
  openArtifact: (ref: ArtifactRef) => void;
  statusByPath: Map<string, string>;
  loadChildren: (path: string) => void;
}

const FileTreeContext = createContext<FileTreeContextValue | null>(null);

function useFileTreeContext() {
  const ctx = useContext(FileTreeContext);
  if (!ctx) throw new Error("FileTreeRow must be used within FileTree");
  return ctx;
}

function FileGlyph({ name }: { name: string }) {
  return <FileTypeIcon name={name} size={16} />;
}

function FileTreeRow({ node, style }: NodeRendererProps<SandboxTreeNode>) {
  const {
    selectedFilePath,
    openFile,
    openArtifact,
    statusByPath,
    loadChildren,
  } = useFileTreeContext();
  const status = statusByPath.get(node.data.id);

  function handleClick() {
    if (node.data.artifact) {
      openArtifact(node.data.artifact);
      return;
    }
    if (node.data.isDir) {
      if (!node.isOpen) loadChildren(node.data.id);
      node.toggle();
    } else {
      openFile(node.data.id);
    }
  }

  return (
    <div
      style={style}
      onClick={handleClick}
      className={cn(
        "flex h-full cursor-pointer items-center gap-1 rounded-lg px-1 text-sm text-zinc-700 hover:bg-zinc-100",
        selectedFilePath === node.data.id && "bg-neutral-100",
      )}
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
          <FileTypeIcon
            name={node.data.name}
            isDir
            isOpen={node.isOpen}
            size={16}
          />
        </>
      ) : (
        <>
          <span className="w-3 shrink-0" />
          <FileGlyph name={node.data.name} />
        </>
      )}
      <span
        className={cn(
          "truncate",
          selectedFilePath === node.data.id && "font-medium text-zinc-900",
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

interface Props extends FileTreeContextValue {
  treeData: SandboxTreeNode[];
}

export function FileTree({ treeData, ...ctx }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      setSize({ width: element.clientWidth, height: element.clientHeight });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <FileTreeContext.Provider value={ctx}>
      <div className="flex h-full flex-col p-2">
        <div className="relative mb-2">
          <MagnifyingGlassIcon
            size={14}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400"
          />
          <input
            type="text"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="Search files"
            aria-label="Search files"
            className="w-full rounded-full border border-zinc-200 bg-white py-1 pl-8 pr-3 text-sm text-zinc-700 [corner-shape:squircle] placeholder:text-zinc-400 focus:border-zinc-300 focus:outline-none"
          />
        </div>
        <div ref={containerRef} className="min-h-0 flex-1 overflow-hidden">
          <Tree
            data={treeData}
            searchTerm={searchTerm}
            searchMatch={(node, term) =>
              node.data.name.toLowerCase().includes(term.toLowerCase())
            }
            width={size.width || 240}
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
      </div>
    </FileTreeContext.Provider>
  );
}
