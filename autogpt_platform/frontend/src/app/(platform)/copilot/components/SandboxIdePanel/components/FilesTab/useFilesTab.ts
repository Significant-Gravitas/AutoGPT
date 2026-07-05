import {
  getGetV2GetSandboxTreeQueryOptions,
  useGetV2GetSandboxChanges,
  useGetV2GetSandboxTree,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SandboxTreeEntry } from "@/app/api/__generated__/models/sandboxTreeEntry";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useCopilotUIStore } from "../../../../store";

export interface SandboxTreeNode {
  id: string;
  name: string;
  isDir: boolean;
  children?: SandboxTreeNode[];
}

function toNodes(entries: SandboxTreeEntry[]): SandboxTreeNode[] {
  return entries.map((entry) =>
    entry.type === "dir"
      ? { id: entry.path, name: entry.name, isDir: true, children: [] }
      : { id: entry.path, name: entry.name, isDir: false },
  );
}

function replaceChildren(
  nodes: SandboxTreeNode[],
  targetId: string,
  children: SandboxTreeNode[],
): SandboxTreeNode[] {
  return nodes.map((node) => {
    if (node.id === targetId) return { ...node, children };
    if (node.children) {
      return {
        ...node,
        children: replaceChildren(node.children, targetId, children),
      };
    }
    return node;
  });
}

export function useFilesTab(sessionId: string) {
  const queryClient = useQueryClient();
  const selectedFilePath = useCopilotUIStore(
    (s) => s.sandboxIdePanel.selectedFilePath,
  );
  const selectFile = useCopilotUIStore((s) => s.selectSandboxFile);

  const { data: rootEntries, isLoading } = useGetV2GetSandboxTree(
    sessionId,
    { path: "" },
    { query: { select: (res) => res.data.entries } },
  );
  const { data: changedFiles } = useGetV2GetSandboxChanges(sessionId, {
    query: { select: (res) => res.data.files },
  });

  const [treeData, setTreeData] = useState<SandboxTreeNode[]>([]);
  const loadedPaths = useRef<Set<string>>(new Set([""]));

  useEffect(() => {
    if (rootEntries) setTreeData(toNodes(rootEntries));
  }, [rootEntries]);

  const statusByPath = new Map<string, string>();
  for (const file of changedFiles ?? []) statusByPath.set(file.path, file.status);

  async function loadChildren(path: string) {
    if (loadedPaths.current.has(path)) return;
    loadedPaths.current.add(path);
    const res = await queryClient.fetchQuery(
      getGetV2GetSandboxTreeQueryOptions(sessionId, { path }),
    );
    setTreeData((prev) => replaceChildren(prev, path, toNodes(res.data.entries)));
  }

  return {
    treeData,
    isLoading,
    selectedFilePath,
    selectFile,
    statusByPath,
    loadChildren,
  };
}
