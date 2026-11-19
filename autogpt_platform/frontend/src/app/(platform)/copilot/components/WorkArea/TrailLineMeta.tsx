import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import type { TaskActivity } from "@/app/api/__generated__/models/taskActivity";
import type { TaskActivityAccount } from "@/app/api/__generated__/models/taskActivityAccount";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";

const CHIP =
  "inline-flex min-w-0 items-center gap-1 rounded-sm bg-foreground-ink/5 px-1";

interface Props {
  line: TaskActivity | null;
}

/** Which connected account and which skill a line's step used, as quiet
 *  chips beside it; nothing when neither is known. */
export function TrailLineMeta({ line }: Props) {
  const account = line?.account;
  const skill = line?.skill;
  if (!account && !skill) return null;
  return (
    <span className="ml-1.5 flex max-w-[50%] min-w-0 shrink items-center gap-1">
      {account ? (
        <Chip kind="Account" text={accountText(account)}>
          <IntegrationLogo
            provider={account.service}
            alt=""
            size={12}
            className="shrink-0"
          />
        </Chip>
      ) : null}
      {skill ? (
        <Chip kind="Skill" text={skill}>
          <Icon
            icon={BookOpen01Icon}
            size={12}
            tone="faint"
            className="shrink-0"
          />
        </Chip>
      ) : null}
    </span>
  );
}

interface ChipProps {
  kind: string;
  text: string;
  children: ReactNode;
}

function Chip({ kind, text, children }: ChipProps) {
  return (
    <span className={CHIP} title={`${kind}: ${text}`}>
      {children}
      <Text as="span" variant="caption-sm" tone="quiet" className="truncate">
        <span className="sr-only">{kind}: </span>
        {text}
      </Text>
    </span>
  );
}

/** "Gmail · work@acme.com", or "Gmail" for an account with no name of its own. */
function accountText(account: TaskActivityAccount) {
  return account.label ? `${account.name} · ${account.label}` : account.name;
}
