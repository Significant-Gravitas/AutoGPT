import { useCallback, useState } from "react";
import type { LeafPane, PaneNode, SplitDirection } from "./types";

let nextPaneId = 1;

function createLeaf(sessionId: string | null = null): LeafPane {
  return { type: "leaf", id: `pane-${nextPaneId++}`, sessionId };
}

/** Replace a node in the tree by id, returning a new tree. */
function replaceNode(
  tree: PaneNode,
  targetId: string,
  replacement: PaneNode,
): PaneNode {
  if (tree.id === targetId) return replacement;
  if (tree.type === "leaf") return tree;
  return {
    ...tree,
    children: [
      replaceNode(tree.children[0], targetId, replacement),
      replaceNode(tree.children[1], targetId, replacement),
    ],
  };
}

/** Remove a leaf from the tree, promoting its sibling. */
function removeLeaf(tree: PaneNode, targetId: string): PaneNode | null {
  if (tree.type === "leaf") {
    return tree.id === targetId ? null : tree;
  }

  const [left, right] = tree.children;

  // If the target is a direct child, return the sibling
  if (left.id === targetId) return right;
  if (right.id === targetId) return left;

  // Recurse into children
  const newLeft = removeLeaf(left, targetId);
  const newRight = removeLeaf(right, targetId);

  if (!newLeft) return newRight;
  if (!newRight) return newLeft;

  return { ...tree, children: [newLeft, newRight] };
}

/** Count leaf nodes in the tree. */
function countLeaves(tree: PaneNode): number {
  if (tree.type === "leaf") return 1;
  return countLeaves(tree.children[0]) + countLeaves(tree.children[1]);
}

/** Update the sessionId of a specific leaf pane. */
function updateLeafSession(
  tree: PaneNode,
  paneId: string,
  sessionId: string | null,
): PaneNode {
  if (tree.type === "leaf") {
    if (tree.id === paneId) return { ...tree, sessionId };
    return tree;
  }
  return {
    ...tree,
    children: [
      updateLeafSession(tree.children[0], paneId, sessionId),
      updateLeafSession(tree.children[1], paneId, sessionId),
    ],
  };
}

export function usePaneTree() {
  const [tree, setTree] = useState<PaneNode>(() => createLeaf());
  const [focusedPaneId, setFocusedPaneId] = useState<string>(
    () => (tree as LeafPane).id,
  );

  const splitPane = useCallback((paneId: string, direction: SplitDirection) => {
    setTree((prev) => {
      // Find the target leaf to preserve its session
      const targetLeaf = findLeaf(prev, paneId);
      const existingLeaf: LeafPane = targetLeaf
        ? { ...targetLeaf }
        : createLeaf();
      // Give the existing leaf a new id so React re-keys properly
      existingLeaf.id = `pane-${nextPaneId++}`;

      const newLeaf = createLeaf();

      const splitNode: PaneNode = {
        type: "split",
        id: `split-${nextPaneId++}`,
        direction,
        children: [existingLeaf, newLeaf],
      };

      setFocusedPaneId(newLeaf.id);
      return replaceNode(prev, paneId, splitNode);
    });
  }, []);

  const closePane = useCallback(
    (paneId: string) => {
      setTree((prev) => {
        if (countLeaves(prev) <= 1) return prev; // Don't close the last pane
        const newTree = removeLeaf(prev, paneId);
        if (!newTree) return prev;

        // If the focused pane was closed, focus the first available leaf
        if (focusedPaneId === paneId) {
          const firstLeaf = findFirstLeaf(newTree);
          if (firstLeaf) setFocusedPaneId(firstLeaf.id);
        }

        return newTree;
      });
    },
    [focusedPaneId],
  );

  const setPaneSession = useCallback(
    (paneId: string, sessionId: string | null) => {
      setTree((prev) => updateLeafSession(prev, paneId, sessionId));
    },
    [],
  );

  return {
    tree,
    focusedPaneId,
    setFocusedPaneId,
    splitPane,
    closePane,
    setPaneSession,
    leafCount: countLeaves(tree),
  };
}

function findLeaf(tree: PaneNode, id: string): LeafPane | null {
  if (tree.type === "leaf") return tree.id === id ? tree : null;
  return findLeaf(tree.children[0], id) || findLeaf(tree.children[1], id);
}

function findFirstLeaf(tree: PaneNode): LeafPane | null {
  if (tree.type === "leaf") return tree;
  return findFirstLeaf(tree.children[0]);
}
