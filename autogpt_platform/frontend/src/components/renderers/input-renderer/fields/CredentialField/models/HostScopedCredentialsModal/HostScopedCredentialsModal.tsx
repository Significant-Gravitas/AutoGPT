import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormDescription,
  FormField,
  FormLabel,
} from "@/components/__legacy__/ui/form";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import { useHostScopedCredentialsModal } from "./useHostScopedCredentialsModal";
import { toDisplayName } from "../../helpers";
import { GlobeIcon, PlusIcon, TrashIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";

type Props = {
  schema: BlockIOCredentialsSubSchema;
  provider: string;
  discriminatorValue?: string;
};

export function HostScopedCredentialsModal({
  schema,
  provider,
  discriminatorValue,
}: Props) {
  const {
    form,
    schemaDescription,
    onSubmit,
    isOpen,
    setIsOpen,
    headerPairs,
    addHeaderPair,
    removeHeaderPair,
    updateHeaderPair,
    currentHost,
  } = useHostScopedCredentialsModal({ schema, provider, discriminatorValue });

  return (
    <>
      <Dialog
        title={`Add sensitive headers for ${toDisplayName(provider) ?? ""}`}
        controlled={{
          isOpen: isOpen,
          set: (isOpen) => {
            if (!isOpen) setIsOpen(false);
          },
        }}
        onClose={() => setIsOpen(false)}
        styling={{
          maxWidth: "38rem",
        }}
      >
        <Dialog.Content>
          <div className="px-1">
            {schemaDescription && (
              <p className="mb-4 text-sm text-zinc-600">{schemaDescription}</p>
            )}

            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="space-y-4"
              >
                <FormField
                  control={form.control}
                  name="host"
                  render={({ field }) => (
                    <Input
                      id="host"
                      label="Host Pattern"
                      type="text"
                      size="small"
                      readOnly={!!currentHost}
                      hint={
                        currentHost
                          ? "Auto-populated from the URL field. Headers will be applied to requests to this host."
                          : "Enter the host/domain to match against request URLs (e.g., api.example.com)."
                      }
                      placeholder={
                        currentHost
                          ? undefined
                          : "Enter host (e.g., api.example.com)"
                      }
                      {...field}
                    />
                  )}
                />

                <FormField
                  control={form.control}
                  name="title"
                  render={({ field }) => (
                    <Input
                      id="title"
                      label="Name (optional)"
                      type="text"
                      placeholder="Enter a name for these credentials..."
                      size="small"
                      {...field}
                    />
                  )}
                />

                <div className="space-y-2">
                  <FormLabel>Headers</FormLabel>
                  <FormDescription className="max-w-md">
                    Add sensitive headers (like Authorization, X-API-Key) that
                    should be automatically included in requests to the
                    specified host.
                  </FormDescription>

                  {headerPairs.map((pair, index) => (
                    <div key={index} className="flex w-full items-center gap-2">
                      <Input
                        id={`header-${index}-key`}
                        label="Header Name"
                        placeholder="e.g., Authorization"
                        size="small"
                        value={pair.key}
                        className="flex-1"
                        onChange={(e) =>
                          updateHeaderPair(index, "key", e.target.value)
                        }
                      />

                      <Input
                        id={`header-${index}-value`}
                        label="Header Value"
                        size="small"
                        type="password"
                        className="flex-1"
                        placeholder="e.g., Bearer token123"
                        value={pair.value}
                        onChange={(e) =>
                          updateHeaderPair(index, "value", e.target.value)
                        }
                      />
                      <Button
                        type="button"
                        variant="secondary"
                        size="small"
                        onClick={() => removeHeaderPair(index)}
                        disabled={headerPairs.length === 1}
                        className="min-w-0"
                      >
                        <TrashIcon className="size-4" />
                      </Button>
                    </div>
                  ))}

                  <Button
                    type="button"
                    variant="outline"
                    size="small"
                    onClick={addHeaderPair}
                  >
                    <PlusIcon className="size-4" /> Add Another Header
                  </Button>
                </div>

                <Button type="submit" size="small" className="min-w-68">
                  Save & use these credentials
                </Button>
              </form>
            </Form>
          </div>
        </Dialog.Content>
      </Dialog>
      <Button
        type="button"
        className="w-fit px-2"
        size="small"
        onClick={() => setIsOpen(true)}
      >
        <GlobeIcon />
        <Text variant="small" className="truncate !text-white opacity-100">
          Add sensitive headers for{" "}
          {toDisplayName(discriminatorValue || provider) ?? ""}
        </Text>
      </Button>
    </>
  );
}
