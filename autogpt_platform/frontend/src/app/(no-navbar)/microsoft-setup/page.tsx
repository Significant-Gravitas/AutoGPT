import Link from "next/link";
import { WorkIQInstructions } from "@/components/contextual/MicrosoftConnection/WorkIQInstructions";

export default function MicrosoftSetupPage() {
  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 p-6">
      <h1 className="text-2xl font-semibold">
        Set up Microsoft 365 for your organization
      </h1>
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold">Approve AutoGPT once</h2>
        <p>
          People can connect their own mail, calendar and files. Teams chats,
          Microsoft 365 Copilot and Work IQ usually need a Microsoft admin to
          approve them first.
        </p>
        <ol className="list-decimal space-y-2 pl-5 text-sm">
          <li>Sign in to AutoGPT with your work account.</li>
          <li>
            Open{" "}
            <Link href="/settings/integrations" className="underline">
              Settings → Integrations
            </Link>{" "}
            and connect Microsoft.
          </li>
          <li>
            Choose <strong>Approve for my whole organization</strong> and accept
            Microsoft&apos;s permissions screen.
          </li>
        </ol>
        <p className="text-sm">
          After that, everyone in your organization gets these services in one
          click.
        </p>
      </section>
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold">Work IQ (optional)</h2>
        <WorkIQInstructions />
      </section>
      <Link href="/settings/integrations" className="underline">
        Return to connections
      </Link>
    </main>
  );
}
