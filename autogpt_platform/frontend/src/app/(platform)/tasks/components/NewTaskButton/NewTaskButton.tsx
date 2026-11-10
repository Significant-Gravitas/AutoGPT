import { Button } from "@/components/atoms/Button/Button";
import { PlusSignIcon } from "@hugeicons/core-free-icons";

interface Props {
  onClick: () => void;
}

export function NewTaskButton({ onClick }: Props) {
  return (
    <Button
      variant="primary"
      size="small"
      leadingIcon={PlusSignIcon}
      onClick={onClick}
    >
      New task
    </Button>
  );
}
