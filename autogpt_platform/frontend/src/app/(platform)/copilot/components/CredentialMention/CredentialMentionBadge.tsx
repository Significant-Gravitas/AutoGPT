import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";

interface Props {
  name: string;
  provider: string;
}

export function CredentialMentionBadge({ name, provider }: Props) {
  return (
    <span className="mx-0.5 inline-flex max-w-full items-center gap-1.5 rounded-xl border border-blue-200 bg-blue-50 px-1.5 py-0.5 align-baseline text-sm font-medium leading-5 text-blue-900">
      <IntegrationLogo
        provider={provider === "codex" ? "openai" : provider}
        alt=""
        size={14}
        className="shrink-0"
      />
      <span className="truncate">{name}</span>
    </span>
  );
}
