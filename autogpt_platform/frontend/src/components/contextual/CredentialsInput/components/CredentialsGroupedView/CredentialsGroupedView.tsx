import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import {
  CredentialsMetaInput,
  CredentialsType,
} from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { SlidersHorizontalIcon } from "@phosphor-icons/react";
import { useContext, useEffect, useMemo, useRef } from "react";
import {
  areSystemCredentialProvidersLoading,
  CredentialField,
  findSavedCredentialByProviderAndType,
  hasMissingRequiredSystemCredentials,
  splitCredentialFieldsBySystem,
} from "./helpers";

type Props = {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
  inputCredentials: Record<string, CredentialsMetaInput | undefined>;
  inputValues: Record<string, any>;
  onCredentialChange: (key: string, value?: CredentialsMetaInput) => void;
};

export function CredentialsGroupedView({
  credentialFields,
  requiredCredentials,
  inputCredentials,
  inputValues,
  onCredentialChange,
}: Props) {
  const allProviders = useContext(CredentialsProvidersContext);

  const { userCredentialFields, systemCredentialFields } = useMemo(
    () => splitCredentialFieldsBySystem(credentialFields, allProviders),
    [credentialFields, allProviders],
  );

  const hasSystemCredentials = systemCredentialFields.length > 0;
  const hasUserCredentials = userCredentialFields.length > 0;
  const hasAttemptedAutoSelect = useRef(false);

  const isLoadingProviders = useMemo(
    () =>
      areSystemCredentialProvidersLoading(systemCredentialFields, allProviders),
    [systemCredentialFields, allProviders],
  );

  const hasMissingSystemCredentials = useMemo(() => {
    if (isLoadingProviders) return false;
    return hasMissingRequiredSystemCredentials(
      systemCredentialFields,
      requiredCredentials,
      inputCredentials,
      allProviders,
    );
  }, [
    isLoadingProviders,
    systemCredentialFields,
    requiredCredentials,
    inputCredentials,
    allProviders,
  ]);

  useEffect(() => {
    if (hasAttemptedAutoSelect.current) return;
    if (!hasSystemCredentials) return;
    if (isLoadingProviders) return;

    for (const [key, schema] of systemCredentialFields) {
      const alreadySelected = inputCredentials?.[key];
      const isRequired = requiredCredentials.has(key);
      if (alreadySelected || !isRequired) continue;

      const providerNames = schema.credentials_provider || [];
      const credentialTypes = schema.credentials_types || [];
      const requiredScopes = schema.credentials_scopes;
      const discriminatorValues = schema.discriminator_values;
      const savedCredential = findSavedCredentialByProviderAndType(
        providerNames,
        credentialTypes,
        requiredScopes,
        allProviders,
        discriminatorValues,
      );

      if (savedCredential) {
        onCredentialChange(key, {
          id: savedCredential.id,
          provider: savedCredential.provider,
          type: savedCredential.type as CredentialsType,
          title: savedCredential.title,
        });
      }
    }

    hasAttemptedAutoSelect.current = true;
  }, [
    allProviders,
    hasSystemCredentials,
    systemCredentialFields,
    requiredCredentials,
    inputCredentials,
    onCredentialChange,
    isLoadingProviders,
  ]);

  return (
    <div className="space-y-6">
      {hasUserCredentials && (
        <>
          {userCredentialFields.map(
            ([key, inputSubSchema]: CredentialField) => {
              const selectedCred = inputCredentials?.[key];

              return (
                <CredentialsInput
                  key={key}
                  schema={
                    { ...inputSubSchema, discriminator: undefined } as any
                  }
                  selectedCredentials={selectedCred}
                  onSelectCredentials={(value) => {
                    onCredentialChange(key, value);
                  }}
                  siblingInputs={inputValues}
                  isOptional={!requiredCredentials.has(key)}
                />
              );
            },
          )}
        </>
      )}

      {hasSystemCredentials && (
        <Accordion
          type="single"
          collapsible
          className={hasUserCredentials ? "mt-4" : ""}
        >
          <AccordionItem value="system-credentials" className="border-none">
            <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
              <div className="flex items-center gap-1">
                <SlidersHorizontalIcon size={16} weight="bold" /> System
                credentials
                {hasMissingSystemCredentials && (
                  <span className="text-destructive">(missing)</span>
                )}
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-6 px-1 pt-2">
                {systemCredentialFields.map(
                  ([key, inputSubSchema]: CredentialField) => {
                    const selectedCred = inputCredentials?.[key];

                    return (
                      <CredentialsInput
                        key={key}
                        schema={
                          { ...inputSubSchema, discriminator: undefined } as any
                        }
                        selectedCredentials={selectedCred}
                        onSelectCredentials={(value) => {
                          onCredentialChange(key, value);
                        }}
                        siblingInputs={inputValues}
                        isOptional={!requiredCredentials.has(key)}
                      />
                    );
                  },
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}
