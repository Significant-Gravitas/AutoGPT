import { useCopyPasteStore } from "@/app/(platform)/build/stores/copyPasteStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  DropdownMenu,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  SecondaryDropdownMenuContent,
  SecondaryDropdownMenuItem,
  SecondaryDropdownMenuSeparator,
} from "@/components/molecules/SecondaryMenu/SecondaryMenu";
import {
  ArrowSquareOutIcon,
  CopyIcon,
  DotsThreeOutlineVerticalIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import { useReactFlow } from "@xyflow/react";

type Props = {
  nodeId: string;
  subGraphID?: string;
};

export const NodeContextMenu = ({ nodeId, subGraphID }: Props) => {
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
      <SecondaryDropdownMenuContent side="right" align="start">
        <SecondaryDropdownMenuItem onClick={handleCopy}>
          <CopyIcon size={20} className="mr-2 dark:text-gray-100" />
          <span className="dark:text-gray-100">Copy</span>
        </SecondaryDropdownMenuItem>
        <SecondaryDropdownMenuSeparator />

        {subGraphID && (
          <>
            <SecondaryDropdownMenuItem
              onClick={() => window.open(`/build?flowID=${subGraphID}`)}
            >
              <ArrowSquareOutIcon
                size={20}
                className="mr-2 dark:text-gray-100"
              />
              <span className="dark:text-gray-100">Open agent</span>
            </SecondaryDropdownMenuItem>
            <SecondaryDropdownMenuSeparator />
          </>
        )}

        <SecondaryDropdownMenuItem variant="destructive" onClick={handleDelete}>
          <TrashIcon
            size={20}
            className="mr-2 text-red-500 dark:text-red-400"
          />
          <span className="dark:text-red-400">Delete</span>
        </SecondaryDropdownMenuItem>
      </SecondaryDropdownMenuContent>
    </DropdownMenu>
  );
};
