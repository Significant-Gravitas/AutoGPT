import { useCopyPasteStore } from "@/app/(platform)/build/stores/copyPasteStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import * as ContextMenu from "@radix-ui/react-context-menu";
import { CopyIcon, TrashIcon } from "@radix-ui/react-icons";
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

  const ContextMenuContent = () => (
    <ContextMenu.Content className="z-10 rounded-xl border bg-white p-1 shadow-md dark:bg-gray-800">
      <ContextMenu.Item
        onSelect={copyNode}
        className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
      >
        <CopyIcon className="mr-2 h-5 w-5 dark:text-gray-100" />
        <span className="dark:text-gray-100">Copy</span>
      </ContextMenu.Item>
      <ContextMenu.Separator className="my-1 h-px bg-gray-300 dark:bg-gray-600" />
      <ContextMenu.Item
        onSelect={deleteNode}
        className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700"
      >
        <TrashIcon className="mr-2 h-5 w-5 text-red-500 dark:text-red-400" />
        <span className="dark:text-red-400">Delete</span>
      </ContextMenu.Item>
    </ContextMenu.Content>
  );

  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger asChild>{children}</ContextMenu.Trigger>
      <ContextMenuContent />
    </ContextMenu.Root>
  );
}
