"use client";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
} from "@/components/__legacy__/ui/form";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import { User } from "@supabase/supabase-js";
import { useNotificationForm } from "./useNotificationForm";
import { Switch } from "@/components/atoms/Switch/Switch";

type NotificationFormProps = {
  preferences: NotificationPreference;
  user: User;
};

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
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="mt-6 flex flex-col gap-10"
        >
          {/* Agent Notifications */}
          <div className="flex flex-col gap-6">
            <Text variant="h4" size="body-medium" className="text-slate-400">
              Agent Notifications
            </Text>
            <FormField
              control={form.control}
              name="notifyOnAgentRun"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Agent Run Notifications
                    </Text>
                    <Text variant="body">
                      Receive notifications when an agent starts or completes a
                      run
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notifyOnBlockExecutionFailed"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Block Execution Failures
                    </Text>
                    <Text variant="body">
                      Get notified when a block execution fails during agent
                      runs
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notifyOnContinuousAgentError"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Continuous Agent Errors
                    </Text>
                    <Text variant="body">
                      Receive alerts when an agent encounters repeated errors
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
          </div>

          {/* Store Notifications */}
          <div className="flex flex-col gap-6">
            <Text variant="h4" size="body-medium" className="text-slate-400">
              Store Notifications
            </Text>
            <FormField
              control={form.control}
              name="notifyOnAgentApproved"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Agent Approved
                    </Text>
                    <Text variant="body">
                      Get notified when your submitted agent is approved for the
                      store
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notifyOnAgentRejected"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Agent Rejected
                    </Text>
                    <Text variant="body">
                      Receive notifications when your agent submission needs
                      updates
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
          </div>

          {/* Balance Notifications */}
          <div className="flex flex-col gap-4">
            <Text variant="h4" size="body-medium" className="text-slate-400">
              Balance Notifications
            </Text>
            <FormField
              control={form.control}
              name="notifyOnZeroBalance"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Zero Balance Alert
                    </Text>
                    <Text variant="body">
                      Get notified when your account balance reaches zero
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notifyOnLowBalance"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Low Balance Warning
                    </Text>
                    <Text variant="body">
                      Receive warnings when your balance is running low
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
          </div>

          {/* Summary Reports */}
          <div className="flex flex-col gap-4">
            <Text variant="h4" size="body-medium" className="text-slate-400">
              Summary reports
            </Text>
            <FormField
              control={form.control}
              name="notifyOnDailySummary"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-1">
                    <Text variant="h4" size="body-medium">
                      Daily Summary
                    </Text>
                    <Text variant="body">
                      Receive a daily summary of your account activity
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notifyOnWeeklySummary"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Weekly Summary
                    </Text>
                    <Text variant="body">
                      Get a weekly overview of your account performance
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notifyOnMonthlySummary"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Text variant="h4" size="body-medium">
                      Monthly Summary
                    </Text>
                    <Text variant="body">
                      Receive a comprehensive monthly report of your account
                    </Text>
                  </div>
                  <FormControl>
                    <Switch
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
