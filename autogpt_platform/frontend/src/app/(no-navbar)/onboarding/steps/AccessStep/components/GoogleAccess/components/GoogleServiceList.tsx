import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { GOOGLE_SERVICES, type GoogleService, isAllowed } from "../helpers";

interface Props {
  /** The connected account; before one, the list only says what each is for. */
  credential?: CredentialsMetaResponse;
  pending: boolean;
  onAllow: (service: GoogleService) => void;
}

export function GoogleServiceList({ credential, pending, onAllow }: Props) {
  return (
    <ul
      aria-label="Google services"
      aria-live="polite"
      className="flex flex-col divide-y divide-separator rounded-lg border border-border-base"
    >
      {GOOGLE_SERVICES.map((service) => {
        const allowed = isAllowed(service, credential?.scopes);
        return (
          <li
            key={service.name}
            className="flex items-center justify-between gap-3 px-3 py-2"
          >
            <div className="flex min-w-0 flex-col">
              <Text variant="body-medium">{service.name}</Text>
              <Text variant="small" tone="muted">
                {service.purpose}
              </Text>
              {credential && (
                <Text variant="small" tone="body">
                  {allowed ? "Ready" : "Needs your permission"}
                </Text>
              )}
            </div>
            {credential && !allowed && (
              <Button
                size="small"
                variant="secondary"
                aria-label={`Allow ${service.name}`}
                disabled={pending}
                onClick={() => onAllow(service)}
              >
                Allow
              </Button>
            )}
          </li>
        );
      })}
    </ul>
  );
}
