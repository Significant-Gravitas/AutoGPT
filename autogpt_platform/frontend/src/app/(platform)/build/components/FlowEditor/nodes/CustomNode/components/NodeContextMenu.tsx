import { useCopyPasteStore } from "@/app/(platform)/build/stores/copyPasteStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { DotsThreeOutlineVerticalIcon } from "@phosphor-icons/react";
import { CopyIcon, ExitIcon, TrashIcon } from "@radix-ui/react-icons";
import { useReactFlow } from "@xyflow/react";

export const NodeContextMenu = ({
  nodeId,
  subGraphID,
}: {
  nodeId: string;
  subGraphID?: string;
}) => {
  const { deleteElements } = useReactFlow();

  function handleCopy() {
    useNodeStore.setState((state) => ({
      nodes: state.nodes.map((node) => ({
        ...node,
        selected: node.id === nodeId,
      })),
    }));

    useCopyPasteStore.getState().copySelectedNodes();
    useCopyPasteStore.getState().pasteNodes();
  }

  function handleDelete() {
    deleteElements({ nodes: [{ id: nodeId }] });
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="py-2">
        <DotsThreeOutlineVerticalIcon size={16} weight="fill" />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="right"
        align="start"
        className="z-10 rounded-xl border bg-white p-1 shadow-md dark:bg-gray-800"
      >
        <DropdownMenuItem
          onClick={handleCopy}
          className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <CopyIcon className="mr-2 h-5 w-5 dark:text-gray-100" />
          <span className="dark:text-gray-100">Copy</span>
        </DropdownMenuItem>

        {subGraphID && (
          <DropdownMenuItem
            onClick={() => window.open(`/build?flowID=${subGraphID}`)}
            className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <ExitIcon className="mr-2 h-5 w-5 dark:text-gray-100" />
            <span className="dark:text-gray-100">Open agent</span>
          </DropdownMenuItem>
        )}

        <div className="my-1 h-px bg-gray-300 dark:bg-gray-600" />

        <DropdownMenuItem
          onClick={handleDelete}
          className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <TrashIcon className="mr-2 h-5 w-5 text-red-500 dark:text-red-400" />
          <span className="dark:text-red-400">Delete</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
