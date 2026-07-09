import {
  getGetV2GetSandboxTreeQueryOptions,
  useGetV2GetSandboxChanges,
  useGetV2GetSandboxTree,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SandboxChangesResponse } from "@/app/api/__generated__/models/sandboxChangesResponse";
import type { SandboxTreeEntry } from "@/app/api/__generated__/models/sandboxTreeEntry";
import type { SandboxTreeResponse } from "@/app/api/__generated__/models/sandboxTreeResponse";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { fileItemToArtifactRef } from "../../../ContextPanel/components/FilesTab/helpers";
import { useSessionFiles } from "../../../ContextPanel/components/FilesTab/useSessionFiles";
import type { ArtifactRef } from "../../../../store";
import { useCopilotUIStore } from "../../../../store";

// Synthetic root folders that group the two file sources shown in the tree.
export const ARTIFACTS_ROOT_ID = "__artifacts__";
export const SANDBOX_ROOT_ID = "__sandbox__";
const ARTIFACT_NODE_PREFIX = "__artifact__:";

export interface SandboxTreeNode {
  id: string;
  name: string;
  isDir: boolean;
  children?: SandboxTreeNode[];
  /** Present on nodes under the Artifacts root — clicking opens the artifact
   *  preview instead of the sandbox editor. */
  artifact?: ArtifactRef;
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
  const openFilePaths = useCopilotUIStore(
    (s) => s.sandboxIdePanel.openFilePaths,
  );
  const selectFile = useCopilotUIStore((s) => s.selectSandboxFile);
  const openFile = useCopilotUIStore((s) => s.openSandboxFile);
  const closeFile = useCopilotUIStore((s) => s.closeSandboxFile);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);

  const { data: rootEntries, isLoading } = useGetV2GetSandboxTree(
    sessionId,
    { path: "" },
    {
      query: {
        select: (res) => (res.data as SandboxTreeResponse).entries,
      },
    },
  );
  const { data: changedFiles } = useGetV2GetSandboxChanges(sessionId, {
    query: {
      select: (res) => (res.data as SandboxChangesResponse).files,
    },
  });
  const { generated, uploaded } = useSessionFiles(sessionId);

  const [sandboxTree, setSandboxTree] = useState<SandboxTreeNode[]>([]);
  const loadedPaths = useRef<Set<string>>(new Set([""]));

  useEffect(() => {
    if (rootEntries) setSandboxTree(toNodes(rootEntries));
  }, [rootEntries]);

  const statusByPath = new Map<string, string>();
  for (const file of changedFiles ?? [])
    statusByPath.set(file.path, file.status);

  const artifactNodes: SandboxTreeNode[] = [...generated, ...uploaded].map(
    (file) => {
      const ref = fileItemToArtifactRef(file.item);
      return {
        id: `${ARTIFACT_NODE_PREFIX}${ref.id}`,
        name: ref.title,
        isDir: false,
        artifact: ref,
      };
    },
  );

  // Two root folders: generated/uploaded artifacts and the live sandbox
  // workspace. The sandbox subtree keeps lazy-loading (its dirs use real
  // paths); the synthetic roots and artifact leaves never fetch.
  const treeData: SandboxTreeNode[] = [
    {
      id: ARTIFACTS_ROOT_ID,
      name: "Artifacts",
      isDir: true,
      children: artifactNodes,
    },
    {
      id: SANDBOX_ROOT_ID,
      name: "Sandbox",
      isDir: true,
      children: sandboxTree,
    },
  ];

  async function loadChildren(path: string) {
    if (
      path === ARTIFACTS_ROOT_ID ||
      path === SANDBOX_ROOT_ID ||
      path.startsWith(ARTIFACT_NODE_PREFIX)
    ) {
      return;
    }
    if (loadedPaths.current.has(path)) return;
    loadedPaths.current.add(path);
    const res = await queryClient.fetchQuery(
      getGetV2GetSandboxTreeQueryOptions(sessionId, { path }),
    );
    const entries = (res.data as SandboxTreeResponse).entries;
    setSandboxTree((prev) => replaceChildren(prev, path, toNodes(entries)));
  }

  return {
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
  };
}
