"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { useInlineConnectForm } from "./useInlineConnectForm";

const userPasswordConnectSchema = z
  .object({
    title: z
      .string()
      .trim()
      .min(1, "Name is required")
      .max(100, "Name must be 100 characters or less"),
    username: z.string().trim().min(1, "Username is required"),
    password: z.string().min(1, "Password is required"),
  })
  .strict();

type UserPasswordConnectFormValues = z.infer<typeof userPasswordConnectSchema>;

interface Props {
  provider: string;
  providerName: string;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

// Carries its own submit, because the panel footer's Continue only drives
// OAuth and API key.
export function InlineUserPasswordForm({
  provider,
  providerName,
  onSuccess,
}: Props) {
  const { submit, isPending } = useInlineConnectForm({
    provider,
    successTitle: "Credentials saved",
    failureTitle: "Couldn't save credentials",
    onSuccess,
  });

  const form = useForm<UserPasswordConnectFormValues>({
    resolver: zodResolver(userPasswordConnectSchema),
    defaultValues: { title: "", username: "", password: "" },
    mode: "onChange",
  });

  function handleSubmit(values: UserPasswordConnectFormValues) {
    submit({
      provider,
      type: "user_password",
      title: values.title,
      username: values.username,
      password: values.password,
    });
  }

  return (
    <Form form={form} onSubmit={handleSubmit} className="space-y-2.5">
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
                placeholder={`My ${providerName} account`}
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="username"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={field.name}
                autoComplete="off"
                spellCheck={false}
                label="Username"
                labelVariant="small-medium"
                size="small"
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="password"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={field.name}
                type="password"
                autoComplete="new-password"
                label="Password"
                labelVariant="small-medium"
                size="small"
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
        size="small"
        className="w-full"
        disabled={!form.formState.isValid}
        loading={isPending}
      >
        {isPending ? "Connecting…" : "Connect"}
      </Button>
    </Form>
  );
}
