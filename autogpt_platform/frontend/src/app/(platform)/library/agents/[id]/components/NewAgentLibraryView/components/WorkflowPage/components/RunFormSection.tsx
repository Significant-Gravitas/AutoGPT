import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { PageField } from "@/app/api/__generated__/models/pageField";
import type { PageRunForm } from "@/app/api/__generated__/models/pageRunForm";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import { useState } from "react";
import { RunAgentInputs } from "../../modals/RunAgentInputs/RunAgentInputs";
import { useAgentRunModal } from "../../modals/RunAgentModal/useAgentRunModal";
import { defaultValues, groupFields } from "../helpers";

interface Props {
  section: PageRunForm;
  agent?: LibraryAgent;
  onRunStarted?: (run: GraphExecutionMeta) => void;
}

export function RunFormSection({ section, agent, onRunStarted }: Props) {
  if (!agent) return <PreviewForm section={section} />;
  if (agent.trigger_setup_info) {
    return (
      <Text variant="body" tone="secondary">
        This workflow starts from a trigger. Set it up from the library to run
        it.
      </Text>
    );
  }
  return (
    <LiveForm section={section} agent={agent} onRunStarted={onRunStarted} />
  );
}

function LiveForm({
  section,
  agent,
  onRunStarted,
}: Props & { agent: LibraryAgent }) {
  // Stable on purpose: the hook resets the form whenever this reference changes.
  const [initialInputValues] = useState(() => defaultValues(section.fields));
  const run = useAgentRunModal(agent, {
    onRun: onRunStarted,
    initialInputValues,
  });
  const credentialFields = Object.entries(
    run.agentCredentialsInputFields ?? {},
  );

  function setInput(key: string, value: unknown) {
    run.setInputValues((prev) => ({ ...prev, [key]: value }));
  }

  function setCredential(key: string, value: unknown) {
    run.setInputCredentials((prev) => {
      const next = { ...prev };
      if (value === undefined) delete next[key];
      else next[key] = value;
      return next;
    });
  }

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        run.handleRun();
      }}
    >
      <Fields
        fields={section.fields}
        values={run.inputValues}
        onChange={setInput}
      />
      {credentialFields.length ? (
        <CredentialsGroupedView
          credentialFields={credentialFields}
          requiredCredentials={
            new Set(
              (agent.credentials_input_schema?.required as string[]) ?? [],
            )
          }
          inputCredentials={run.inputCredentials}
          inputValues={run.inputValues}
          onCredentialChange={setCredential}
        />
      ) : null}
      <div>
        <Button type="submit" size="small" loading={run.isExecuting}>
          {section.button_label}
        </Button>
      </div>
    </form>
  );
}

function PreviewForm({ section }: { section: PageRunForm }) {
  const values = defaultValues(section.fields);
  return (
    <div className="flex flex-col gap-4">
      <Fields fields={section.fields} values={values} readOnly />
      <div>
        <Button size="small" disabled>
          {section.button_label}
        </Button>
      </div>
    </div>
  );
}

interface FieldsProps {
  fields: PageField[];
  values: Record<string, unknown>;
  onChange?: (key: string, value: unknown) => void;
  readOnly?: boolean;
}

function Fields({ fields, values, onChange, readOnly }: FieldsProps) {
  return (
    <>
      {groupFields(fields).map((group, index) => (
        <fieldset key={index} className="flex min-w-0 flex-col gap-3">
          {group.name ? (
            <legend className="mb-1">
              <Text as="span" variant="body-medium">
                {group.name}
              </Text>
            </legend>
          ) : null}
          {group.fields.map((field) => (
            <RunAgentInputs
              key={field.input}
              schema={fieldSchema(field)}
              value={values[field.input]}
              placeholder={field.help ?? undefined}
              onChange={(value) => onChange?.(field.input, value)}
              readOnly={readOnly}
              data-testid={`page-input-${field.input}`}
            />
          ))}
        </fieldset>
      ))}
    </>
  );
}

function fieldSchema(field: PageField): BlockIOSubSchema {
  return {
    ...field.input_schema,
    title: field.label,
    description: field.help ?? field.input_schema.description,
  } as BlockIOSubSchema;
}
