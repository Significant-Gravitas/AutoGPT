import type { MicrosoftCapabilityService } from "@/app/api/__generated__/models/microsoftCapabilityService";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { WorkIQInstructions } from "@/components/contextual/MicrosoftConnection/WorkIQInstructions";
import { MICROSOFT_SERVICE_PURPOSES } from "@/lib/microsoft-setup";

const SERVICES: Array<{ service: MicrosoftCapabilityService; name: string }> = [
  { service: "mail", name: "Mail" },
  { service: "calendar", name: "Calendar" },
  { service: "files", name: "Files" },
  { service: "teams", name: "Teams chats" },
  { service: "copilot", name: "Microsoft 365 Copilot" },
  { service: "work_iq", name: "Work IQ" },
];

const ADMIN_CONSENT_GUIDE_URL =
  "https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/grant-admin-consent";

export default function MicrosoftSetupPage() {
  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 p-6">
      <Text variant="h2" as="h1">
        Set up Microsoft 365 for your organization
      </Text>
      <section className="flex flex-col gap-3">
        <Text variant="h4" as="h2">
          What AutoGPT asks for
        </Text>
        <Text variant="body">
          Each person connects their own Microsoft account, and each service
          asks Microsoft for its own permissions.
        </Text>
        <ul className="flex flex-col gap-2">
          {SERVICES.map(({ service, name }) => (
            <li key={service} className="flex flex-col">
              <Text variant="body-medium">{name}</Text>
              <Text variant="small" tone="muted">
                {MICROSOFT_SERVICE_PURPOSES[service]}
              </Text>
            </li>
          ))}
        </ul>
      </section>
      <section className="flex flex-col gap-3">
        <Text variant="h4" as="h2">
          Approve AutoGPT once
        </Text>
        <Text variant="body">
          People can usually approve mail, calendar and files themselves. Teams
          chats and Microsoft 365 Copilot usually need a Microsoft admin, and
          Work IQ always does.
        </Text>
        <ol className="list-decimal space-y-2 pl-5">
          <Text as="li" variant="body">
            Sign in to AutoGPT with your work account.
          </Text>
          <Text as="li" variant="body">
            Open{" "}
            <Link href="/settings/integrations" variant="secondary">
              Settings → Integrations
            </Link>{" "}
            and connect Microsoft.
          </Text>
          <Text as="li" variant="body">
            If Settings offers{" "}
            <strong>Approve for my whole organization</strong>, choose it and
            accept Microsoft&apos;s permissions screen. If it doesn&apos;t, this
            AutoGPT server connects Microsoft with sign-in codes: grant AutoGPT
            tenant-wide admin consent by following{" "}
            <Link href={ADMIN_CONSENT_GUIDE_URL} isExternal variant="secondary">
              Microsoft&apos;s guide
            </Link>
            .
          </Text>
        </ol>
        <Text variant="body">
          After that, people in your organization can connect without waiting
          for an admin.
        </Text>
      </section>
      <section className="flex flex-col gap-3">
        <Text variant="h4" as="h2">
          Work IQ (optional)
        </Text>
        <WorkIQInstructions />
      </section>
      <Link href="/settings/integrations" variant="secondary">
        Return to connections
      </Link>
    </main>
  );
}
