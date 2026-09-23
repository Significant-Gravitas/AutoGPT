"use client";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
} from "@/components/__legacy__/ui/form";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import type { User } from "@/lib/auth/types";
import {
  useNotificationForm,
  type NotificationFormValues,
} from "./useNotificationForm";
import { Switch } from "@/components/atoms/Switch/Switch";

type NotificationFormProps = {
  preferences: NotificationPreference;
  user: User;
};

const BRIEFING_OPTIONS = [
  { value: "DAILY", label: "Daily" },
  { value: "WEEKLY", label: "Weekly" },
  { value: "MONTHLY", label: "Monthly" },
  { value: "OFF", label: "Off" },
];

export function NotificationForm({ preferences, user }: NotificationFormProps) {
  const { form, onSubmit, onCancel, isLoading } = useNotificationForm({
    preferences,
    user,
  });

  return (
    <div>
      <Text variant="h3" size="large-semibold">
        Notifications
      </Text>
      <Text variant="body" className="mt-2 text-slate-400">
        Billing and account messages are always sent — they are about your
        account, not a promotion.
      </Text>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="mt-6 flex flex-col gap-10"
        >
          <div className="flex flex-col gap-6">
            <FormField
              control={form.control}
              name="briefingFrequency"
              render={({ field }) => (
                <FormItem className="flex flex-col gap-2">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Briefing
                    </Text>
                    <Text variant="body">
                      What your agents got done, at around 07:30 your time.
                      Never sent when nothing ran.
                    </Text>
                  </div>
                  <FormControl>
                    <Select
                      id="briefing-frequency"
                      label="Briefing frequency"
                      hideLabel
                      value={field.value}
                      options={BRIEFING_OPTIONS}
                      onValueChange={(value) =>
                        field.onChange(
                          value as NotificationFormValues["briefingFrequency"],
                        )
                      }
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="alertsEnabled"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Alerts
                    </Text>
                    <Text variant="body">
                      Only when something is waiting on you — never for a
                      successful run. At most two a day.
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      aria-label="Alerts"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="storeVerdictsEnabled"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Marketplace reviews
                    </Text>
                    <Text variant="body">
                      When an agent you submitted is approved or needs changes.
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      aria-label="Marketplace reviews"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
          </div>

          {/* Form Actions */}
          <div className="flex justify-end gap-4 pt-8">
            <Button
              variant="outline"
              type="button"
              onClick={onCancel}
              disabled={isLoading}
              className="min-w-[10rem]"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading || !form.formState.isDirty}
              className="min-w-[10rem]"
              loading={isLoading}
            >
              {isLoading ? "Saving..." : "Save preferences"}
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
