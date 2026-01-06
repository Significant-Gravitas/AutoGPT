import { useCopyPasteStore } from "@/app/(platform)/build/stores/copyPasteStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  SecondaryMenuContent,
  SecondaryMenuItem,
  SecondaryMenuSeparator,
} from "@/components/molecules/SecondaryMenu/SecondaryMenu";
import { CopyIcon, TrashIcon } from "@phosphor-icons/react";
import * as ContextMenu from "@radix-ui/react-context-menu";
import { useReactFlow } from "@xyflow/react";
import { CustomNode } from "../CustomNode";

export function NodeRightClickMenu({
  nodeId,
  children,
}: {
  nodeId: string;
  children: React.ReactNode;
}) {
  const { deleteElements } = useReactFlow<CustomNode>();

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

  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger asChild>
        <div>{children}</div>
      </ContextMenu.Trigger>
      <SecondaryMenuContent>
        <SecondaryMenuItem onSelect={copyNode}>
          <CopyIcon size={20} className="mr-2 dark:text-gray-100" />
          <span className="dark:text-gray-100">Copy</span>
        </SecondaryMenuItem>
        <SecondaryMenuSeparator />
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
