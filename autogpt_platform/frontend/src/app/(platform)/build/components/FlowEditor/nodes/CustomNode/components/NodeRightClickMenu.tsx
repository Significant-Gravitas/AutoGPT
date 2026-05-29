import { useCopyPasteStore } from "@/app/(platform)/build/stores/copyPasteStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  SecondaryMenuContent,
  SecondaryMenuItem,
  SecondaryMenuSeparator,
} from "@/components/molecules/SecondaryMenu/SecondaryMenu";
import { ArrowSquareOutIcon, CopyIcon, TrashIcon } from "@phosphor-icons/react";
import * as ContextMenu from "@radix-ui/react-context-menu";
import { useReactFlow } from "@xyflow/react";
import { useEffect, useRef } from "react";
import { CustomNode } from "../CustomNode";

type Props = {
  nodeId: string;
  subGraphID?: string;
  children: React.ReactNode;
};

const DOUBLE_CLICK_TIMEOUT = 300;

export function NodeRightClickMenu({ nodeId, subGraphID, children }: Props) {
  const { deleteElements } = useReactFlow<CustomNode>();
  const lastRightClickTime = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);

  function copyNode() {
    useNodeStore.setState((state) => ({
      nodes: state.nodes.map((node) => ({
        ...node,
        selected: node.id === nodeId,
      })),
    }));

    useCopyPasteStore.getState().copySelectedNodes();
    useCopyPasteStore.getState().pasteNodes();
  }

  function deleteNode() {
    deleteElements({ nodes: [{ id: nodeId }] });
  }

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    function handleContextMenu(e: MouseEvent) {
      const now = Date.now();
      const timeSinceLastClick = now - lastRightClickTime.current;

      if (timeSinceLastClick < DOUBLE_CLICK_TIMEOUT) {
        e.stopImmediatePropagation();
        lastRightClickTime.current = 0;
        return;
      }

      lastRightClickTime.current = now;
    }

    container.addEventListener("contextmenu", handleContextMenu, true);

    return () => {
      container.removeEventListener("contextmenu", handleContextMenu, true);
    };
  }, []);

  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger asChild>
        <div ref={containerRef}>{children}</div>
      </ContextMenu.Trigger>
      <SecondaryMenuContent>
        <SecondaryMenuItem onSelect={copyNode}>
          <CopyIcon size={20} className="mr-2 dark:text-gray-100" />
          <span className="dark:text-gray-100">Copy</span>
        </SecondaryMenuItem>
        <SecondaryMenuSeparator />

        {subGraphID && (
          <>
            <SecondaryMenuItem
              onClick={() => window.open(`/build?flowID=${subGraphID}`)}
            >
              <ArrowSquareOutIcon
                size={20}
                className="mr-2 dark:text-gray-100"
              />
              <span className="dark:text-gray-100">Open agent</span>
            </SecondaryMenuItem>
            <SecondaryMenuSeparator />
          </>
        )}

        <SecondaryMenuItem variant="destructive" onSelect={deleteNode}>
          <TrashIcon
            size={20}
            className="mr-2 text-red-500 dark:text-red-400"
          />
          <span className="dark:text-red-400">Delete</span>
        </SecondaryMenuItem>
      </SecondaryMenuContent>
    </ContextMenu.Root>
  );
}
