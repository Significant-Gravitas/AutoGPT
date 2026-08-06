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
import { useReactFlow } from "@xyflow/react";
import {
  Copy01Icon,
  Delete02Icon,
  LinkSquare01Icon,
  MoreVerticalIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
        <Icon icon={MoreVerticalIcon} size={16} />
      </DropdownMenuTrigger>
      <SecondaryDropdownMenuContent side="right" align="start">
        <SecondaryDropdownMenuItem onClick={handleCopy}>
          <Icon
            icon={Copy01Icon}
            size={20}
            className="mr-2 dark:text-gray-100"
          />
          <span className="dark:text-gray-100">Copy</span>
        </SecondaryDropdownMenuItem>
        <SecondaryDropdownMenuSeparator />

        {subGraphID && (
          <>
            <SecondaryDropdownMenuItem
              onClick={() => window.open(`/build?flowID=${subGraphID}`)}
            >
              <Icon
                icon={LinkSquare01Icon}
                size={20}
                className="mr-2 dark:text-gray-100"
              />
              <span className="dark:text-gray-100">Open agent</span>
            </SecondaryDropdownMenuItem>
            <SecondaryDropdownMenuSeparator />
          </>
        )}

        <SecondaryDropdownMenuItem variant="destructive" onClick={handleDelete}>
          <Icon
            icon={Delete02Icon}
            size={20}
            className="mr-2 text-red-500 dark:text-red-400"
          />
          <span className="dark:text-red-400">Delete</span>
        </SecondaryDropdownMenuItem>
      </SecondaryDropdownMenuContent>
    </DropdownMenu>
  );
};
