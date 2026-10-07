import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
interface Props {
  name: string;
  onContinue: () => void;
  isSwitching?: boolean;
}
export function ProviderContinuation({ name, onContinue, isSwitching }: Props) {
  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-zinc-200 pt-4">
      <Text variant="small" className="!text-zinc-500">
        Your connected account is also available.
      </Text>
      <Button
        size="small"
        variant="ghost"
        onClick={onContinue}
        loading={isSwitching}
        rightIcon={<Icon icon={ArrowRight02Icon} size={14} />}
      >
        Continue on {name}
      </Button>
    </div>
  );
}
