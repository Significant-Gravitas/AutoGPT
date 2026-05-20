"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { useApiKeyConnectForm } from "./useApiKeyConnectForm";

interface Props {
  provider: string;
  providerName: string;
  onSuccess: () => void;
}

export function ApiKeyConnectForm({
  provider,
  providerName,
  onSuccess,
}: Props) {
  const { form, handleSubmit, isPending } = useApiKeyConnectForm({
    provider,
    onSuccess,
  });

  return (
    <Form form={form} onSubmit={handleSubmit} className="flex flex-col gap-4">
      <FormField
        control={form.control}
        name="title"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={field.name}
                autoComplete="off"
                label="Name"
                placeholder={`My ${providerName} key`}
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="apiKey"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={field.name}
                type="password"
                autoComplete="new-password"
                spellCheck={false}
                label="API key"
                placeholder="sk-..."
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="expiresAt"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                value={field.value ?? ""}
                id={field.name}
                type="date"
                label="Expires (optional)"
                hint="Leave blank to keep the key indefinitely"
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <Button
        type="submit"
        variant="primary"
        size="large"
        disabled={!form.formState.isValid || isPending}
        loading={isPending}
      >
        Save API key
      </Button>
    </Form>
  );
}
