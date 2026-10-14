import { Button } from "@/components/atoms/Button/Button";
import { PlusSignIcon } from "@hugeicons/core-free-icons";

interface Props {
  onAdd: () => void;
}

export function AddColumnButton({ onAdd }: Props) {
  return (
    <Button
      variant="ghost"
      size="small"
      leadingIcon={PlusSignIcon}
      onClick={onAdd}
      className="w-72 shrink-0 justify-start rounded-xl text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800"
    >
      Add column
    </Button>
  );
}
