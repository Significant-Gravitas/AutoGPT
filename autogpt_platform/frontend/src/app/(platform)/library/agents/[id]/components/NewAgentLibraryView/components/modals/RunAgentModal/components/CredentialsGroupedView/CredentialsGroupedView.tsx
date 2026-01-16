import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { SlidersHorizontal } from "@phosphor-icons/react";
import { useContext, useEffect, useMemo, useRef } from "react";
import { useRunAgentModalContext } from "../../context";
import {
  areSystemCredentialProvidersLoading,
  CredentialField,
  findSavedCredentialByProviderAndType,
  hasMissingRequiredSystemCredentials,
  splitCredentialFieldsBySystem,
} from "../helpers";

type Props = {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
};

export function CredentialsGroupedView({
  credentialFields,
  requiredCredentials,
}: Props) {
  const allProviders = useContext(CredentialsProvidersContext);
  const { inputCredentials, setInputCredentialsValue, inputValues } =
    useRunAgentModalContext();

  const { userCredentialFields, systemCredentialFields } = useMemo(
    () =>
      splitCredentialFieldsBySystem(
        credentialFields,
        allProviders,
        inputCredentials,
      ),
    [credentialFields, allProviders, inputCredentials],
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
      const savedCredential = findSavedCredentialByProviderAndType(
        providerNames,
        credentialTypes,
        requiredScopes,
        allProviders,
      );

      if (savedCredential) {
        setInputCredentialsValue(key, {
          id: savedCredential.id,
          provider: savedCredential.provider,
          type: savedCredential.type,
          title: (savedCredential as { title?: string }).title,
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
    setInputCredentialsValue,
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
                    setInputCredentialsValue(key, value);
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
                <SlidersHorizontal size={16} weight="bold" /> System credentials
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
                          setInputCredentialsValue(key, value);
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
