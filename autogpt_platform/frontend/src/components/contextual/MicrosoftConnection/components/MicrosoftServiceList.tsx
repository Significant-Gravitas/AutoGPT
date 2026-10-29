import type { MicrosoftAccessCheck } from "@/app/api/__generated__/models/microsoftAccessCheck";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { MICROSOFT_STATUS_LABELS } from "@/lib/microsoft-setup";
import { canAllow } from "../helpers";

interface Props {
  check?: MicrosoftAccessCheck;
  checking: boolean;
  browserAvailable: boolean;
  onAllow: (scopes: string[]) => void;
}

export function MicrosoftServiceList({
  check,
  checking,
  browserAvailable,
  onAllow,
}: Props) {
  if (!check) {
    return (
      <Text variant="small" tone="muted" role="status">
        {checking
          ? "Checking what this account can use…"
          : "Couldn't check this account. Try again in a moment."}
      </Text>
    );
  }

  return (
    <ul
      aria-label="Microsoft services"
      className="flex flex-col divide-y divide-separator rounded-lg border border-border-base"
    >
      {check.capabilities.map((item) => (
        <li
          key={item.service}
          className="flex items-center justify-between gap-3 px-3 py-2"
        >
          <div className="flex min-w-0 flex-col">
            <Text variant="body-medium">{item.name}</Text>
            <Text variant="small" tone="muted">
              {MICROSOFT_STATUS_LABELS[item.status]}
            </Text>
          </div>
          {browserAvailable && canAllow(item) ? (
            <Button
              size="small"
              variant="secondary"
              aria-label={`Allow ${item.name}`}
              onClick={() => onAllow(item.scopes ?? [])}
            >
              Allow
            </Button>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
