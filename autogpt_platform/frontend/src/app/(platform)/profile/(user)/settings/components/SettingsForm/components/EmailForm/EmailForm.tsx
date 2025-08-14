"use client";

import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { User } from "@supabase/supabase-js";
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
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="mt-6 flex flex-col gap-4"
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
              className="min-w-[10rem]"
            >
              Reset password
            </Button>
            <Button
              type="submit"
              disabled={hasError || isSameEmail}
              loading={isLoading}
              className="min-w-[10rem]"
            >
              {isLoading ? "Saving..." : "Update email"}
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
