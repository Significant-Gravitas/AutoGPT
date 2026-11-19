import { Text } from "@/components/atoms/Text/Text";
import type { CSSProperties } from "react";
import type { RecipientOption } from "../../../../ChatInput/components/RecipientChip";
import { getAvatarGlow } from "../../../helpers";
import { RecipientAvatar } from "../../RecipientAvatar";

interface Props {
  recipient: RecipientOption;
  role: string;
  description: string | null;
}

export function InfoHeader({ recipient, role, description }: Props) {
  return (
    <header className="flex flex-col gap-3">
      <span
        className="inline-flex self-start rounded-2xl shadow-(--avatar-glow)"
        style={
          { "--avatar-glow": getAvatarGlow(recipient, 24) } as CSSProperties
        }
      >
        <RecipientAvatar recipient={recipient} size={56} shape="tile" />
      </span>
      <div className="flex min-w-0 flex-col gap-1">
        <div className="flex min-w-0 items-center gap-2">
          <Text variant="lead-semibold" as="h2" tone="ink" className="truncate">
            {recipient.name}
          </Text>
          <span className="inline-flex h-5 shrink-0 items-center rounded-control bg-surface-wash px-1.5">
            <Text variant="small" weight="medium" as="span" tone="text">
              {role}
            </Text>
          </span>
        </div>
        {description && (
          <Text variant="inbox" as="p" tone="quiet" className="line-clamp-2">
            {description}
          </Text>
        )}
      </div>
    </header>
  );
}
