import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SquareLock02Icon } from "@hugeicons/core-free-icons";

interface Props {
  label: string;
  lockLine: string;
  isProcessing: boolean;
  isBlocked: boolean;
  error?: string | null;
  onClick: () => void;
}

// The step's one action. On narrow screens it stays at the bottom of the
// viewport while the panel it belongs to is on screen.
export function CheckoutAction({
  label,
  lockLine,
  isProcessing,
  isBlocked,
  error,
  onClick,
}: Props) {
  return (
    <div className="sticky bottom-0 z-10 -mx-5 mt-auto flex flex-col gap-2.5 border-t border-border-base bg-surface px-5 pb-4 pt-3 sm:-mx-8 sm:px-8 lg:static lg:mx-0 lg:border-t-0 lg:px-0 lg:pb-0 lg:pt-0">
      <Button
        variant="primary"
        size="large"
        onClick={onClick}
        loading={isProcessing}
        disabled={isProcessing || isBlocked}
        className="h-12 w-full text-base"
      >
        {label}
      </Button>
      {error ? (
        <Text variant="body" role="alert" tone="danger">
          {error}
        </Text>
      ) : null}
      <Text variant="small" tone="muted" className="text-balance text-center">
        <Icon
          icon={SquareLock02Icon}
          size={13}
          className="mr-1 inline-block -translate-y-px align-middle"
        />
        {lockLine}
      </Text>
    </div>
  );
}
