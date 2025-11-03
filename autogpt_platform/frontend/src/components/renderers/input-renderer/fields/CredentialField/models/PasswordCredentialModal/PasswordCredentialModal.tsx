import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import { usePasswordCredentialModal } from "./usePasswordCredentialModal";
import { toDisplayName } from "../../helpers";
import { UserIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";

type Props = {
  provider: string;
};

export function PasswordCredentialsModal({ provider }: Props) {
  const { form, onSubmit, open, setOpen } = usePasswordCredentialModal({
    provider,
  });

  return (
    <>
      <Dialog
        title={`Add new username & password for ${toDisplayName(provider)}`}
        controlled={{
          isOpen: open,
          set: (isOpen) => {
            if (!isOpen) setOpen(false);
          },
        }}
        onClose={() => setOpen(false)}
        styling={{
          maxWidth: "25rem",
        }}
      >
        <Dialog.Content>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="space-y-2 pt-4"
            >
              <FormField
                control={form.control}
                name="username"
                render={({ field }) => (
                  <Input
                    id="username"
                    label="Username"
                    type="text"
                    placeholder="Enter username..."
                    size="small"
                    {...field}
                  />
                )}
              />
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <Input
                    id="password"
                    label="Password"
                    type="password"
                    placeholder="Enter password..."
                    size="small"
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
                    label="Name"
                    type="text"
                    placeholder="Enter a name for this user login..."
                    size="small"
                    {...field}
                  />
                )}
              />
              <Button type="submit" size="small" className="min-w-68">
                Save & use this user login
              </Button>
            </form>
          </Form>
        </Dialog.Content>
      </Dialog>
      <Button
        type="button"
        className="w-fit"
        size="small"
        onClick={() => setOpen(true)}
      >
        <UserIcon className="size-4" />
        <Text variant="small" className="!text-white opacity-100">
          Add username & password
        </Text>
      </Button>
    </>
  );
}
