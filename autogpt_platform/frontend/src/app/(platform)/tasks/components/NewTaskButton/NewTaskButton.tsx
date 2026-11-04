import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  onClick: () => void;
}

export function NewTaskButton({ onClick }: Props) {
  return (
    <Button
      variant="primary"
      size="small"
      leftIcon={<Icon icon={PlusSignIcon} size={16} />}
      onClick={onClick}
      aria-keyshortcuts="N"
    >
      New task
    </Button>
  );
}
