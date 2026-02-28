"use client";

import { Group, Panel, Separator } from "react-resizable-panels";
import { ChatPane } from "./ChatPane";
import type { PaneNode } from "./types";

interface Props {
  node: PaneNode;
}

export function PaneTree({ node }: Props) {
  if (node.type === "leaf") {
    return <ChatPane paneId={node.id} sessionId={node.sessionId} />;
  }

  const orientation =
    node.direction === "horizontal" ? "horizontal" : "vertical";

  return (
    <Group orientation={orientation} id={node.id}>
      <Panel defaultSize={50} minSize={15} id={`${node.id}-left`}>
        <PaneTree node={node.children[0]} />
      </Panel>

      <Separator
        className={
          "relative flex items-center justify-center bg-zinc-200 transition-colors hover:bg-violet-300 active:bg-violet-400 " +
          (orientation === "horizontal" ? "w-1.5" : "h-1.5")
        }
      >
        <div
          className={
            "rounded-full bg-zinc-400 " +
            (orientation === "horizontal" ? "h-8 w-1" : "h-1 w-8")
          }
        />
      </Separator>

      <Panel defaultSize={50} minSize={15} id={`${node.id}-right`}>
        <PaneTree node={node.children[1]} />
      </Panel>
    </Group>
  );
}
