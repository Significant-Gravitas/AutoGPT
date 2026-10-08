"use client";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
} from "@/components/molecules/Form/Form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import type { User } from "@/lib/auth/types";
import { useEmailForm } from "./useEmailForm";

type EmailFormProps = {
  user: User;
};

export function EmailForm({ user }: EmailFormProps) {
  const { form, onSubmit, isLoading, currentEmail } = useEmailForm({ user });

  const hasError = Object.keys(form.formState.errors).length > 0;
  const isSameEmail = form.watch("email") === currentEmail;

  return (
    <div>
      <Text variant="h3" size="large-semibold">
        Security & Access
      </Text>
      <Form
        form={form}
        onSubmit={onSubmit}
        className="mt-4 flex flex-col gap-0 space-y-0"
      >
        <FormField
          control={form.control}
          name="email"
          render={({ field, fieldState }) => (
            <FormItem>
              <FormControl>
                <Input
                  id={field.name}
                  label="Email"
                  placeholder="m@example.com"
                  type="text"
                  autoComplete="off"
                  className="w-full"
                  size="md"
                  error={fieldState.error?.message}
                  {...field}
                />
              </FormControl>
            </FormItem>
          )}
        />
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            as="NextLink"
            href="/reset-password"
            className="min-w-40"
            size="md"
          >
            Reset password
          </Button>
          <Button
            type="submit"
            disabled={hasError || isSameEmail}
            loading={isLoading}
            className="min-w-40"
            size="md"
          >
            {isLoading ? "Saving..." : "Update email"}
          </Button>
        </div>
      </Form>
    </div>
  );
}
