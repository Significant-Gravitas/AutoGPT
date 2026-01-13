import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { SlidersHorizontalIcon } from "lucide-react";
import { useContext, useMemo } from "react";
import { useRunAgentModalContext } from "../../context";

type CredentialField = [string, any];

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

  const { userCredentialFields, systemCredentialFields } = useMemo(() => {
    if (!allProviders || credentialFields.length === 0) {
      return { userCredentialFields: [], systemCredentialFields: [] };
    }

    const userFields: CredentialField[] = [];
    const systemFields: CredentialField[] = [];

    for (const [key, schema] of credentialFields) {
      const providerNames = schema.credentials_provider || [];
      const isSystemField = providerNames.some((providerName: string) => {
        const providerData = allProviders[providerName];
        return providerData?.isSystemProvider === true;
      });

      if (isSystemField) {
        systemFields.push([key, schema]);
      } else {
        userFields.push([key, schema]);
      }
    }

    const sortByUnsetFirst = (a: CredentialField, b: CredentialField) => {
      const aIsSet = !!inputCredentials?.[a[0]];
      const bIsSet = !!inputCredentials?.[b[0]];

      if (aIsSet === bIsSet) return 0;
      return aIsSet ? 1 : -1;
    };

    return {
      userCredentialFields: userFields.sort(sortByUnsetFirst),
      systemCredentialFields: systemFields.sort(sortByUnsetFirst),
    };
  }, [credentialFields, allProviders, inputCredentials]);

  const hasSystemCredentials = systemCredentialFields.length > 0;
  const hasUserCredentials = userCredentialFields.length > 0;

  const shouldOpenAccordionByDefault = useMemo(() => {
    if (!hasSystemCredentials) return false;

    return systemCredentialFields.some(([key]) => {
      const isRequired = requiredCredentials.has(key);
      const selectedCred = inputCredentials?.[key];
      return isRequired && !selectedCred;
    });
  }, [
    hasSystemCredentials,
    systemCredentialFields,
    requiredCredentials,
    inputCredentials,
  ]);

  return (
    <div className="space-y-6">
      {hasUserCredentials && (
        <>
          {userCredentialFields.map(([key, inputSubSchema]) => {
            const selectedCred = inputCredentials?.[key];

            return (
              <CredentialsInput
                key={key}
                schema={{ ...inputSubSchema, discriminator: undefined } as any}
                selectedCredentials={selectedCred}
                onSelectCredentials={(value) => {
                  setInputCredentialsValue(key, value);
                }}
                siblingInputs={inputValues}
                isOptional={!requiredCredentials.has(key)}
              />
            );
          })}
        </>
      )}

      {hasSystemCredentials && (
        <Accordion
          type="single"
          collapsible
          defaultValue={
            shouldOpenAccordionByDefault ? "system-credentials" : undefined
          }
          className={hasUserCredentials ? "mt-4" : ""}
        >
          <AccordionItem value="system-credentials" className="border-none">
            <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
              <div className="flex items-center gap-1">
                <SlidersHorizontalIcon className="size-4" /> System credentials
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-6 px-1 pt-2">
                {systemCredentialFields.map(([key, inputSubSchema]) => {
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
                })}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}
