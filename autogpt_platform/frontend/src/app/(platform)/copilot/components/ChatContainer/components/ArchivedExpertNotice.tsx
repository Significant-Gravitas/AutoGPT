import type { ExpertReadOnlyReason } from "@/app/(platform)/copilot/useExpertMap";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { ArchiveIcon, CloudOffIcon } from "@hugeicons/core-free-icons";

type Props = {
  expertName: string;
  reason: ExpertReadOnlyReason;
};

// Only `fired` is a fact we can state. An absent id may have been deleted or
// never have been ours, so it gets neutral copy rather than a firing we can't
// confirm; a failed roster read is explicitly temporary.
const NOTICE_COPY: Record<ExpertReadOnlyReason, (name: string) => string> = {
  fired: (name) => `${name} was fired — this thread is read-only`,
  unavailable: (name) =>
    `${name} is temporarily unavailable — this thread is read-only`,
  unknown: (name) =>
    `${name} is no longer available — this thread is read-only`,
};

export function ArchivedExpertNotice({ expertName, reason }: Props) {
  return (
    <div className="px-3 pb-6 pt-2">
      <AttentionCard.Root
        variant="muted"
        density="relaxed"
        data-testid="archived-expert-notice"
        role="status"
        className="mx-auto w-full max-w-2xl"
      >
        <AttentionCard.Row align="center" justify="center">
          <AttentionCard.Icon>
            <Icon
              icon={reason === "unavailable" ? CloudOffIcon : ArchiveIcon}
              size={16}
            />
          </AttentionCard.Icon>
          <Text variant="small" className="text-muted-foreground">
            {NOTICE_COPY[reason](expertName)}
          </Text>
        </AttentionCard.Row>
      </AttentionCard.Root>
    </div>
  );
}
