import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArchiveIcon, CloudOffIcon } from "@hugeicons/core-free-icons";

interface Props {
  expertName: string;
  isUnavailable?: boolean;
}

export function ArchivedExpertNotice({ expertName, isUnavailable }: Props) {
  return (
    <div className="px-3 pb-6 pt-2">
      <div
        data-testid="archived-expert-notice"
        role="status"
        className="mx-auto flex w-full max-w-2xl items-center justify-center gap-2 rounded-2xl border border-border bg-muted/40 px-4 py-3 text-center"
      >
        <Icon
          icon={isUnavailable ? CloudOffIcon : ArchiveIcon}
          size={16}
          className="text-muted-foreground"
        />
        <Text variant="small" className="text-muted-foreground">
          {isUnavailable
            ? `${expertName} is temporarily unavailable — this thread is read-only`
            : `${expertName} was fired — this thread is read-only`}
        </Text>
      </div>
    </div>
  );
}
