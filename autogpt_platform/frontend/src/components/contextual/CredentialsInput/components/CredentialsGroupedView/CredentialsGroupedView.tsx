import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { CredentialField } from "./helpers";
import { useCredentialsGroupedView } from "./useCredentialsGroupedView";
import { SlidersHorizontalIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const SYSTEM_SECTION = "system-credentials";
const NO_NODE_CREDENTIALS: CredentialsMetaInput[] = [];

type Props = {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
  inputCredentials: Record<string, CredentialsMetaInput | undefined>;
  inputValues: Record<string, any>;
  onCredentialChange: (key: string, value?: CredentialsMetaInput) => void;
  /** The credentials picked on the workflow's steps; a field starts on the user's own key among them. */
  nodeCredentials?: CredentialsMetaInput[];
};

export function CredentialsGroupedView({
  credentialFields,
  requiredCredentials,
  inputCredentials,
  inputValues,
  onCredentialChange,
  nodeCredentials = NO_NODE_CREDENTIALS,
}: Props) {
  const {
    visibleFields,
    collapsedFields,
    hasMissingSystemCredentials,
    handleFieldChange,
    handleSystemSectionChange,
  } = useCredentialsGroupedView({
    credentialFields,
    requiredCredentials,
    inputCredentials,
    onCredentialChange,
    nodeCredentials,
  });

  function renderField([key, inputSubSchema]: CredentialField) {
    return (
      <CredentialsInput
        key={key}
        schema={{ ...inputSubSchema, discriminator: undefined } as any}
        selectedCredentials={inputCredentials?.[key]}
        onSelectCredentials={(value) => {
          handleFieldChange(key, value);
        }}
        siblingInputs={inputValues}
        isOptional={!requiredCredentials.has(key)}
      />
    );
  }

  return (
    <div className="space-y-6">
      {visibleFields.map(renderField)}

      {collapsedFields.length > 0 && (
        <Accordion
          type="single"
          collapsible
          onValueChange={handleSystemSectionChange}
          className={visibleFields.length > 0 ? "mt-4" : ""}
        >
          <AccordionItem value={SYSTEM_SECTION} className="border-none">
            <AccordionTrigger className="py-2 text-body text-foreground-tertiary hover:no-underline">
              <div className="flex items-center gap-1">
                <Icon icon={SlidersHorizontalIcon} size={16} /> System
                credentials
                {hasMissingSystemCredentials && (
                  <span className="text-error">(missing)</span>
                )}
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-6 px-1 pt-2">
                {collapsedFields.map(renderField)}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}
