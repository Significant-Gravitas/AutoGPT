import { Separator } from "@/components/__legacy__/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { DotsThreeOutlineVerticalIcon } from "@phosphor-icons/react";
import { Copy, Trash2, ExternalLink } from "lucide-react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useCopyPasteStore } from "@/app/(platform)/build/stores/copyPasteStore";
import { useReactFlow } from "@xyflow/react";

export const NodeContextMenu = ({
  nodeId,
  subGraphID,
}: {
  nodeId: string;
  subGraphID?: string;
}) => {
  const { deleteElements } = useReactFlow();

  const handleCopy = () => {
    useNodeStore.setState((state) => ({
      nodes: state.nodes.map((node) => ({
        ...node,
        selected: node.id === nodeId,
      })),
    }));

    useCopyPasteStore.getState().copySelectedNodes();
    useCopyPasteStore.getState().pasteNodes();
  };

  const handleDelete = () => {
    deleteElements({ nodes: [{ id: nodeId }] });
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="py-2">
        <DotsThreeOutlineVerticalIcon size={16} weight="fill" />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="right"
        align="start"
        className="rounded-xlarge"
      >
        <DropdownMenuItem onClick={handleCopy} className="hover:rounded-xlarge">
          <Copy className="mr-2 h-4 w-4" />
          Copy Node
        </DropdownMenuItem>

        {subGraphID && (
          <DropdownMenuItem
            onClick={() => window.open(`/build?flowID=${subGraphID}`)}
            className="hover:rounded-xlarge"
          >
            <ExternalLink className="mr-2 h-4 w-4" />
            Open Agent
          </DropdownMenuItem>
        )}

        <Separator className="my-2" />

        <DropdownMenuItem
          onClick={handleDelete}
          className="text-red-600 hover:rounded-xlarge"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
