import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInput";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { SlidersHorizontalIcon } from "lucide-react";
import { useContext, useEffect, useMemo, useRef } from "react";
import { useRunAgentModalContext } from "../../context";
import {
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

  const hasMissingSystemCredentials = useMemo(
    () =>
      hasMissingRequiredSystemCredentials(
        systemCredentialFields,
        requiredCredentials,
        inputCredentials,
      ),
    [systemCredentialFields, requiredCredentials, inputCredentials],
  );

  useEffect(() => {
    if (hasAttemptedAutoSelect.current) return;
    if (!hasSystemCredentials) return;

    let appliedSelection = false;

    for (const [key, schema] of systemCredentialFields) {
      const alreadySelected = inputCredentials?.[key];
      const isRequired = requiredCredentials.has(key);
      if (alreadySelected || !isRequired) continue;

      const providerNames = schema.credentials_provider || [];
      const credentialTypes = schema.credentials_types || [];
      const savedCredential = findSavedCredentialByProviderAndType(
        providerNames,
        credentialTypes,
        allProviders,
      );

      if (savedCredential) {
        appliedSelection = true;
        setInputCredentialsValue(key, {
          id: savedCredential.id,
          provider: savedCredential.provider,
          type: savedCredential.type,
          title: (savedCredential as { title?: string }).title,
        });
      }
    }

    if (appliedSelection) {
      hasAttemptedAutoSelect.current = true;
    }
  }, [
    allProviders,
    hasSystemCredentials,
    systemCredentialFields,
    requiredCredentials,
    inputCredentials,
    setInputCredentialsValue,
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
                <SlidersHorizontalIcon className="size-4" /> System credentials
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
