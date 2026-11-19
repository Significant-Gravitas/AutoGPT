import { FormField } from "@/components/molecules/Form/Form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, FormProvider } from "react-hook-form";
import { z } from "zod";
import {
  buildServerScope,
  getServerFieldSchema,
  getServerScopeSchemes,
} from "../../helpers";

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
  const schemes = credentials
    ? getServerScopeSchemes(credentials.provider)
    : null;

  const formSchema = z.object({
    server:
      schemes && credentials
        ? getServerFieldSchema(credentials.provider)
        : z.string().trim(),
    scheme: z.string(),
    username: z.string().min(1, "Username is required"),
    password: z.string().min(1, "Password is required"),
    title: z.string().min(1, "Name is required"),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      server: "",
      scheme: schemes?.[0].value ?? "",
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
      ...(schemes && { scope: buildServerScope(values.scheme, values.server) }),
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
        <FormProvider {...form}>
          <form
            onSubmit={form.handleSubmit(onSubmit)}
            className="w-[98%] space-y-2 pt-4"
          >
            {schemes && schemes.length > 1 && (
              <FormField
                control={form.control}
                name="scheme"
                render={({ field }) => (
                  <Select
                    id="scheme"
                    label="Database type"
                    value={field.value}
                    onValueChange={field.onChange}
                    options={schemes}
                  />
                )}
              />
            )}
            {schemes && (
              <FormField
                control={form.control}
                name="server"
                render={({ field, fieldState }) => (
                  <Input
                    id="server"
                    error={fieldState.error?.message}
                    label="Server"
                    type="text"
                    placeholder="host or host:port"
                    hint="This login is only ever sent to this server."
                    {...field}
                  />
                )}
              />
            )}
            <FormField
              control={form.control}
              name="username"
              render={({ field, fieldState }) => (
                <Input
                  id="username"
                  error={fieldState.error?.message}
                  label="Username"
                  type="text"
                  placeholder="Enter username..."
                  {...field}
                />
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field, fieldState }) => (
                <Input
                  id="password"
                  error={fieldState.error?.message}
                  label="Password"
                  type="password"
                  placeholder="Enter password..."
                  {...field}
                />
              )}
            />
            <FormField
              control={form.control}
              name="title"
              render={({ field, fieldState }) => (
                <Input
                  id="title"
                  error={fieldState.error?.message}
                  label="Name"
                  type="text"
                  placeholder="Enter a name for this user login..."
                  className="mb-8"
                  {...field}
                />
              )}
            />
            <Button type="submit" className="w-full">
              Save & use this user login
            </Button>
          </form>
        </FormProvider>
      </Dialog.Content>
    </Dialog>
  );
}
