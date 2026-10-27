import {
  WORK_IQ_APP_ID,
  WORK_IQ_PERMISSIONS_URL,
  WORK_IQ_SETUP_URL,
} from "@/lib/microsoft-setup";

export function WorkIQInstructions() {
  return (
    <ol className="list-decimal space-y-3 pl-5 text-sm">
      <li>
        <a
          href={WORK_IQ_SETUP_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline"
        >
          Enable Work IQ for your organization
        </a>
        . This creates Microsoft&apos;s Work IQ service in your tenant.
        <pre className="mt-2 overflow-x-auto rounded bg-zinc-100 p-2 text-xs">
          {`az ad sp create --id ${WORK_IQ_APP_ID}`}
        </pre>
      </li>
      <li>
        Review the{" "}
        <a
          href={WORK_IQ_PERMISSIONS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline"
        >
          Work IQ permission
        </a>
        , then return to Microsoft sign-in to approve it. Company policies may
        also require approval for mail, calendar, files, or Teams.
      </li>
      <li>
        Confirm a Copilot Studio usage-based billing plan is configured and the
        intended user is assigned. Work IQ usage can incur charges.
      </li>
      <li>
        Return here, sign in with the intended account, and verify access.
        Individual actions still follow your organization&apos;s policies.
      </li>
    </ol>
  );
}
