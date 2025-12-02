import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { useState } from "react";
import { getSchemaDefaultCredentials } from "../../helpers";
import { areAllCredentialsSet, getCredentialFields } from "./helpers";

type Credential = CredentialsMetaInput | undefined;
type Credentials = Record<string, Credential>;

type Props = {
  agent: GraphMeta | null;
  siblingInputs?: Record<string, any>;
  onCredentialsChange: (
    credentials: Record<string, CredentialsMetaInput>,
  ) => void;
  onValidationChange: (isValid: boolean) => void;
  onLoadingChange: (isLoading: boolean) => void;
};

export function AgentOnboardingCredentials(props: Props) {
  const [inputCredentials, setInputCredentials] = useState<Credentials>({});

  const fields = getCredentialFields(props.agent);
  const required = Object.keys(fields || {}).length > 0;

  if (!required) return null;

  function handleSelectCredentials(key: string, value: Credential) {
    const updated = { ...inputCredentials, [key]: value };
    setInputCredentials(updated);

    const sanitized: Record<string, CredentialsMetaInput> = {};
    for (const [k, v] of Object.entries(updated)) {
      if (v) sanitized[k] = v;
    }

    props.onCredentialsChange(sanitized);

    const isValid = !required || areAllCredentialsSet(fields, updated);
    props.onValidationChange(isValid);
  }

  return (
    <>
      {Object.entries(fields).map(([key, inputSubSchema]) => (
        <div key={key} className="mt-4">
          <CredentialsInput
            schema={inputSubSchema}
            selectedCredentials={
              inputCredentials[key] ??
              getSchemaDefaultCredentials(inputSubSchema)
            }
            onSelectCredentials={(value) => handleSelectCredentials(key, value)}
            siblingInputs={props.siblingInputs}
            onLoaded={(loaded) => props.onLoadingChange(!loaded)}
          />
        </div>
      ))}
    </>
  );
}
