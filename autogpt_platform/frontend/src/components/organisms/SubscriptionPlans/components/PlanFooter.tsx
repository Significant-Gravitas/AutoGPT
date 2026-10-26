import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onCompare: () => void;
}

export function PlanFooter({ onCompare }: Props) {
  return (
    <footer
      data-slot="plan-footer"
      className="mt-3 flex flex-wrap items-center justify-center gap-x-5 gap-y-2"
    >
      <Text variant="small" tone="muted">
        Need something custom?{" "}
        <a
          href="mailto:sales@agpt.co"
          data-slot="plan-footer-sales-link"
          className="font-medium text-brand no-underline hover:text-brand-hover"
        >
          Talk to sales.
        </a>
      </Text>
      <Button
        type="button"
        variant="link"
        className="h-auto min-w-0 p-0 text-xs text-foreground-tertiary"
        onClick={onCompare}
      >
        Compare plans
      </Button>
    </footer>
  );
}
