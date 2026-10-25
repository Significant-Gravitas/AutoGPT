import { FormProvider } from "react-hook-form";
import { FormDescription, FormField } from "@/components/molecules/Form/Form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { useAPIKeyCredentialsModal } from "./useAPIKeyCredentialsModal";

type Props = {
  schema: BlockIOCredentialsSubSchema;
  open: boolean;
  onClose: () => void;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
};

export function APIKeyCredentialsModal({
  schema,
  open,
  onClose,
  onCredentialsCreate,
  siblingInputs,
}: Props) {
  const {
    form,
    isLoading,
    isSubmitting,
    supportsApiKey,
    providerName,
    schemaDescription,
    onSubmit,
  } = useAPIKeyCredentialsModal({ schema, siblingInputs, onCredentialsCreate });

  if (isLoading || !supportsApiKey) {
    return null;
  }

  return (
    <Dialog
      title={`Add new API key for ${providerName ?? ""}`}
      controlled={{
        isOpen: open,
        set: (isOpen) => {
          if (!isOpen) onClose();
        },
      }}
      onClose={onClose}
      styling={{
        maxWidth: "25rem",
      }}
    >
      <Dialog.Content>
        {schemaDescription && (
          <Text variant="body-tight" as="p" tone="secondary" className="mb-4">
            {schemaDescription}
          </Text>
        )}

        <FormProvider {...form}>
          <form
            onSubmit={form.handleSubmit(onSubmit)}
            className="space-y-2 px-2"
          >
            <FormField
              control={form.control}
              name="title"
              render={({ field }) => (
                <Input
                  id="title"
                  label="Name"
                  type="text"
                  placeholder="Enter a name for this API Key..."
                  {...field}
                />
              )}
            />
            <FormField
              control={form.control}
              name="apiKey"
              render={({ field }) => (
                <>
                  <Input
                    id="apiKey"
                    label="API Key"
                    type="password"
                    placeholder="Enter API Key..."
                    hint={
                      schema.credentials_scopes ? (
                        <FormDescription
                          // TODO(tokens): SECRT-3152 rare size: text-[0.8rem].
                          // dark:text-neutral-400: the legacy FormDescription's, moved verbatim (DS6, SECRT-3116)
                          className="text-[0.8rem] leading-inherit dark:text-neutral-400"
                        >
                          Required scope(s) for this block:{" "}
                          {schema.credentials_scopes?.map((s, i, a) => (
                            <span key={i}>
                              <Text
                                variant="small-mono-bold-tight"
                                as="code"
                                tone="inherit"
                              >
                                {s}
                              </Text>
                              {i < a.length - 1 && ", "}
                            </span>
                          ))}
                        </FormDescription>
                      ) : null
                    }
                    {...field}
                  />
                </>
              )}
            />

            <FormField
              control={form.control}
              name="expiresAt"
              render={({ field }) => (
                <Input
                  id="expiresAt"
                  label="Expiration Date"
                  type="datetime-local"
                  placeholder="Select expiration date..."
                  value={field.value}
                  onChange={(e) => {
                    const value = e.target.value;
                    if (value) {
                      const dateTime = new Date(value);
                      dateTime.setHours(0, 0, 0, 0);
                      const year = dateTime.getFullYear();
                      const month = String(dateTime.getMonth() + 1).padStart(
                        2,
                        "0",
                      );
                      const day = String(dateTime.getDate()).padStart(2, "0");
                      const normalizedValue = `${year}-${month}-${day}T00:00`;
                      field.onChange(normalizedValue);
                    } else {
                      field.onChange(value);
                    }
                  }}
                  onBlur={field.onBlur}
                  name={field.name}
                />
              )}
            />
            <Button
              type="submit"
              className="min-w-68"
              loading={isSubmitting}
              disabled={isSubmitting}
            >
              Add API Key
            </Button>
          </form>
        </FormProvider>
      </Dialog.Content>
    </Dialog>
  );
}
