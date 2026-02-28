/**
 * Pane tree data structures for tmux-style split windowing.
 *
 * The tree is a binary tree where each internal node is a split (horizontal or
 * vertical) and each leaf node is an independent chat pane.
 */

export type SplitDirection = "horizontal" | "vertical";

export interface LeafPane {
  type: "leaf";
  id: string;
  sessionId: string | null;
}

export interface SplitPane {
  type: "split";
  id: string;
  direction: SplitDirection;
  children: [PaneNode, PaneNode];
}

export type PaneNode = LeafPane | SplitPane;
