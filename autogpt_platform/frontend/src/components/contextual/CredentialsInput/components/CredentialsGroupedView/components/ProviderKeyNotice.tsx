import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { Alert02Icon } from "@hugeicons/core-free-icons";

interface Props {
  providers: string[];
  isOptional: boolean;
}

const providerList = new Intl.ListFormat("en", { type: "disjunction" });

// Stands in for the picker of a field open to several providers when the user
// has a key for none of them, so no single provider's keys can be shown
export function ProviderKeyNotice({ providers, isOptional }: Props) {
  return (
    <div className="mb-6">
      <div className="mb-2 flex items-center gap-1">
        <Text variant="small" as="span" tone="secondary">
          Credentials
        </Text>
        {isOptional ? (
          <Text variant="small" as="span" tone="muted" weight="normal">
            (optional)
          </Text>
        ) : (
          <Text
            variant="small"
            as="span"
            tone="error-strong"
            weight="normal"
            className="inline-flex items-center gap-1"
          >
            <Icon aria-hidden icon={Alert02Icon} size={14} />
            required
          </Text>
        )}
      </div>
      <Alert variant="warning">
        <AlertDescription>
          <Text variant="body">
            Add a key for {providerList.format(providers.map(toDisplayName))} in{" "}
            <Link href="/settings/integrations" variant="secondary">
              Settings → Integrations
            </Link>
            .
          </Text>
        </AlertDescription>
      </Alert>
    </div>
  );
}
