"use client";

import { Input } from "@/components/atoms/Input/Input";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";
import type { ApiKeyConnectFormValues } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/schema";
import type { UseFormReturn } from "react-hook-form";

interface Props {
  form: UseFormReturn<ApiKeyConnectFormValues>;
  providerName: string;
  onSubmit: (values: ApiKeyConnectFormValues) => void;
}

// The API-key inputs for the expanded method card: compact spacing and
// small text, and no submit button of its own — the panel footer's
// Continue drives the same submit (Enter still works via the form).
export function InlineApiKeyForm({ form, providerName, onSubmit }: Props) {
  return (
    <Form form={form} onSubmit={onSubmit} className="space-y-2.5">
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
                labelVariant="small-medium"
                size="small"
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
                labelVariant="small-medium"
                size="small"
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
                labelVariant="small-medium"
                size="small"
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
    </Form>
  );
}
