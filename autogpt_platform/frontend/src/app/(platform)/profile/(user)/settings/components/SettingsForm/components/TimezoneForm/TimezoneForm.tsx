"use client";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
} from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import type { User } from "@/lib/auth/types";
import * as React from "react";
import { TIMEZONES } from "./helpers";
import { useTimezoneForm } from "./useTimezoneForm";

type Props = {
  user: User;
  currentTimezone?: string;
};

export function TimezoneForm({ user, currentTimezone = "not-set" }: Props) {
  // If timezone is not set, try to detect it from the browser
  const effectiveTimezone = React.useMemo(() => {
    if (currentTimezone === "not-set") {
      // Try to get browser timezone as a suggestion
      try {
        return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
      } catch {
        return "UTC";
      }
    }
    return currentTimezone;
  }, [currentTimezone]);

  const { form, onSubmit, isLoading } = useTimezoneForm({
    user,
    currentTimezone: effectiveTimezone,
  });

  return (
    <Card>
      <Text variant="h5" as="h3" className="mb-6">
        Timezone
      </Text>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <FormField
            control={form.control}
            name="timezone"
            render={({ field, fieldState }) => (
              <FormItem>
                <FormControl>
                  <Select
                    id="timezone"
                    label="Select your timezone"
                    labelVariant="body-medium"
                    placeholder="Select a timezone"
                    value={field.value}
                    onValueChange={field.onChange}
                    options={TIMEZONES}
                    error={fieldState.error?.message}
                  />
                </FormControl>
              </FormItem>
            )}
          />
          <Button type="submit" disabled={isLoading} size="md">
            {isLoading ? "Saving..." : "Save timezone"}
          </Button>
        </form>
      </Form>
    </Card>
  );
}
