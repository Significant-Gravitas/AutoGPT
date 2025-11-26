import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";

type Props = {
  schema: BlockIOCredentialsSubSchema;
  open: boolean;
  onClose: () => void;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
};

export function PasswordCredentialsModal({
  schema,
  open,
  onClose,
  onCredentialsCreate,
  siblingInputs,
}: Props) {
  const credentials = useCredentials(schema, siblingInputs);

  const formSchema = z.object({
    username: z.string().min(1, "Username is required"),
    password: z.string().min(1, "Password is required"),
    title: z.string().min(1, "Name is required"),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
      password: "",
      title: "",
    },
  });

  if (
    !credentials ||
    credentials.isLoading ||
    !credentials.supportsUserPassword
  ) {
    return null;
  }

  const { provider, providerName, createUserPasswordCredentials } = credentials;

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const newCredentials = await createUserPasswordCredentials({
      username: values.username,
      password: values.password,
      title: values.title,
    });
    onCredentialsCreate({
      provider,
      id: newCredentials.id,
      type: "user_password",
      title: newCredentials.title,
    });
  }

  return (
    <Dialog
      title={`Add new username & password for ${providerName}`}
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
  );
}
