import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { SlidersHorizontalIcon } from "lucide-react";
import { CredentialRow } from "../CredentialRow/CredentialRow";
import { CredentialsSelect } from "../CredentialsSelect/CredentialsSelect";

type Credential = {
  id: string;
  title?: string;
  username?: string;
  type: string;
  provider: string;
};

type Props = {
  schema: BlockIOCredentialsSubSchema;
  provider: string;
  displayName: string;
  userCredentials: Credential[];
  systemCredentials: Credential[];
  isSystemProvider: boolean;
  selectedCredential?: CredentialsMetaInput;
  onSelectCredential: (credentialId: string) => void;
  onClearCredential: () => void;
  onAddCredential: () => void;
  actionButtonText: string;
  isOptional: boolean;
  showTitle: boolean;
  variant: "default" | "node";
};

export function CredentialsAccordionView({
  schema,
  provider,
  displayName,
  userCredentials,
  systemCredentials,
  isSystemProvider,
  selectedCredential,
  onSelectCredential,
  onClearCredential,
  onAddCredential,
  actionButtonText,
  isOptional,
  showTitle,
  variant,
}: Props) {
  const allCredentials = [...userCredentials, ...systemCredentials];
  const hasSystemCredentials = systemCredentials.length > 0;
  const hasUserCredentials = userCredentials.length > 0;

  const credentialsInAccordion = isSystemProvider
    ? allCredentials
    : systemCredentials;

  const shouldOpenAccordionByDefault =
    hasSystemCredentials && !isOptional && !selectedCredential;

  const showUserCredentialsOutsideAccordion =
    !isSystemProvider && hasUserCredentials;

  return (
    <>
      {showTitle && showUserCredentialsOutsideAccordion && (
        <div className="mb-2 flex items-center gap-2">
          <Text variant="large-medium">
            {displayName} credentials
            {isOptional && (
              <span className="ml-1 text-sm font-normal text-gray-500">
                (optional)
              </span>
            )}
          </Text>
          {schema.description && (
            <InformationTooltip description={schema.description} />
          )}
        </div>
      )}

      {showUserCredentialsOutsideAccordion && (
        <>
          {userCredentials.length > 1 || isOptional ? (
            <CredentialsSelect
              credentials={userCredentials}
              provider={provider}
              displayName={displayName}
              selectedCredentials={selectedCredential}
              onSelectCredential={onSelectCredential}
              onClearCredential={onClearCredential}
              allowNone={isOptional}
              variant={variant}
            />
          ) : (
            <div className="mb-4 space-y-2">
              {userCredentials.map((credential) => (
                <CredentialRow
                  key={credential.id}
                  credential={credential}
                  provider={provider}
                  displayName={displayName}
                  onSelect={() => onSelectCredential(credential.id)}
                />
              ))}
            </div>
          )}
          <Button
            variant="secondary"
            size="small"
            onClick={onAddCredential}
            className="w-fit"
            type="button"
          >
            {actionButtonText}
          </Button>
        </>
      )}

      {hasSystemCredentials && (
        <Accordion
          type="single"
          collapsible
          defaultValue={
            shouldOpenAccordionByDefault ? "system-credentials" : undefined
          }
          className={showUserCredentialsOutsideAccordion ? "mt-4" : ""}
        >
          <AccordionItem value="system-credentials" className="border-none">
            <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
              <div className="flex items-center gap-1">
                <SlidersHorizontalIcon className="size-4" /> System credentials
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 px-1 pt-2">
                {showTitle && (
                  <div className="flex items-center gap-2">
                    <Text variant="large-medium">
                      {displayName} credentials
                      {isOptional && (
                        <span className="ml-1 text-sm font-normal text-gray-500">
                          (optional)
                        </span>
                      )}
                    </Text>
                    {schema.description && (
                      <InformationTooltip description={schema.description} />
                    )}
                  </div>
                )}
                {credentialsInAccordion.length > 0 && (
                  <CredentialsSelect
                    credentials={credentialsInAccordion}
                    provider={provider}
                    displayName={displayName}
                    selectedCredentials={selectedCredential}
                    onSelectCredential={onSelectCredential}
                    onClearCredential={onClearCredential}
                    allowNone={isOptional}
                    variant={variant}
                  />
                )}
                {isSystemProvider && (
                  <Button
                    variant="secondary"
                    size="small"
                    onClick={onAddCredential}
                    className="w-fit"
                    type="button"
                  >
                    {actionButtonText}
                  </Button>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {!showUserCredentialsOutsideAccordion && !isSystemProvider && (
        <Button
          variant="secondary"
          size="small"
          onClick={onAddCredential}
          className="w-fit"
          type="button"
        >
          {actionButtonText}
        </Button>
      )}
    </>
  );
}
