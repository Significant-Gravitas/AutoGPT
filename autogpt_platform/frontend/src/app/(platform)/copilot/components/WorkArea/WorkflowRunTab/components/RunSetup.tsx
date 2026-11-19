"use client";

import { RunAgentInputs } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentInputs/RunAgentInputs";
import { useAgentRunModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/useAgentRunModal";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import { ArrowLeft01Icon, PlayIcon } from "@hugeicons/core-free-icons";
import { InfoGroup } from "../../../EmptySession/components/ExpertInfoPanel/components/InfoParts";

interface Props {
  agent: LibraryAgent;
  onStarted: (executionID: string) => void;
  onCancel: () => void;
}

/** What a workflow asks for before it runs, filled in beside the chat: its
 *  inputs and the accounts it uses. It takes the tab's place until the run
 *  starts or the user goes back. */
export function RunSetup({ agent, onStarted, onCancel }: Props) {
  const form = useAgentRunModal(agent, {
    onRun: (execution) => onStarted(execution.id),
  });
  const inputFields = Object.entries(form.agentInputFields);
  const accountFields = Object.entries(form.agentCredentialsInputFields);
  const requiredAccounts = new Set(
    (agent.credentials_input_schema?.required as string[] | undefined) ?? [],
  );

  return (
    <form
      aria-label={`Run ${agent.name}`}
      className="flex flex-col gap-5"
      onSubmit={(event) => {
        event.preventDefault();
        form.handleRun();
      }}
    >
      <div className="flex flex-col gap-1.5 px-1">
        <Button
          type="button"
          variant="ghost-muted"
          size="xs"
          leadingIcon={ArrowLeft01Icon}
          className="-ml-1.5 self-start"
          onClick={onCancel}
        >
          Back to runs
        </Button>
        <Text variant="body-md-medium" as="h2" tone="ink" className="truncate">
          Run {agent.name}
        </Text>
        {agent.description ? (
          <Text
            variant="small"
            as="p"
            tone="quiet"
            className="line-clamp-3 text-pretty"
          >
            {agent.description}
          </Text>
        ) : null}
      </div>

      {inputFields.length > 0 ? (
        <InfoGroup label="Inputs">
          <div className="flex flex-col gap-4 rounded-panel bg-surface p-3 shadow-(--inbox-shadow-card)">
            {inputFields.map(([key, schema]) => (
              <RunAgentInputs
                key={key}
                schema={schema}
                value={form.inputValues[key] ?? schema.default}
                placeholder={schema.description}
                onChange={(value) =>
                  form.setInputValues({ ...form.inputValues, [key]: value })
                }
              />
            ))}
          </div>
        </InfoGroup>
      ) : null}

      {accountFields.length > 0 ? (
        <InfoGroup label="Accounts">
          <div className="rounded-panel bg-surface p-3 shadow-(--inbox-shadow-card)">
            <CredentialsGroupedView
              credentialFields={accountFields}
              requiredCredentials={requiredAccounts}
              inputCredentials={form.inputCredentials}
              inputValues={form.inputValues}
              onCredentialChange={(key, value) =>
                form.setInputCredentials({
                  ...form.inputCredentials,
                  [key]: value,
                })
              }
            />
          </div>
        </InfoGroup>
      ) : null}

      <div className="flex items-center gap-1.5 px-1">
        <Button
          type="submit"
          variant="primary"
          size="xs"
          leadingIcon={PlayIcon}
          loading={form.isExecuting}
        >
          Start run
        </Button>
        <Button type="button" variant="secondary" size="xs" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
