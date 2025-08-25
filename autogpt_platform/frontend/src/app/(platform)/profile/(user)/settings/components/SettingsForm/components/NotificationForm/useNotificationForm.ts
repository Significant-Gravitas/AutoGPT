"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import { User } from "@supabase/supabase-js";
import { usePostV1UpdateNotificationPreferences } from "@/app/api/__generated__/endpoints/auth/auth";
import { NotificationPreferenceDTO } from "@/lib/autogpt-server-api/types";

const notificationFormSchema = z.object({
  notifyOnAgentRun: z.boolean(),
  notifyOnZeroBalance: z.boolean(),
  notifyOnLowBalance: z.boolean(),
  notifyOnBlockExecutionFailed: z.boolean(),
  notifyOnContinuousAgentError: z.boolean(),
  notifyOnDailySummary: z.boolean(),
  notifyOnWeeklySummary: z.boolean(),
  notifyOnMonthlySummary: z.boolean(),
});

function createNotificationDefaultValues(preferences: {
  preferences?: Record<string, boolean>;
}) {
  return {
    notifyOnAgentRun: preferences.preferences?.AGENT_RUN,
    notifyOnZeroBalance: preferences.preferences?.ZERO_BALANCE,
    notifyOnLowBalance: preferences.preferences?.LOW_BALANCE,
    notifyOnBlockExecutionFailed:
      preferences.preferences?.BLOCK_EXECUTION_FAILED,
    notifyOnContinuousAgentError:
      preferences.preferences?.CONTINUOUS_AGENT_ERROR,
    notifyOnDailySummary: preferences.preferences?.DAILY_SUMMARY,
    notifyOnWeeklySummary: preferences.preferences?.WEEKLY_SUMMARY,
    notifyOnMonthlySummary: preferences.preferences?.MONTHLY_SUMMARY,
  };
}

export function useNotificationForm({
  preferences,
  user,
}: {
  preferences: NotificationPreference;
  user: User;
}) {
  const { toast } = useToast();
  const defaultValues = createNotificationDefaultValues(preferences);

  const form = useForm<z.infer<typeof notificationFormSchema>>({
    resolver: zodResolver(notificationFormSchema),
    defaultValues,
  });

  const updateNotificationsMutation = usePostV1UpdateNotificationPreferences({
    mutation: {
      onError: (error) => {
        toast({
          title: "Error updating notifications",
          description:
            error instanceof Error
              ? error.message
              : "Failed to update notification preferences",
          variant: "destructive",
        });
      },
    },
  });

  async function onSubmit(values: z.infer<typeof notificationFormSchema>) {
    try {
      const notificationPreferences: NotificationPreferenceDTO = {
        email: user.email || "",
        preferences: {
          AGENT_RUN: values.notifyOnAgentRun,
          ZERO_BALANCE: values.notifyOnZeroBalance,
          LOW_BALANCE: values.notifyOnLowBalance,
          BLOCK_EXECUTION_FAILED: values.notifyOnBlockExecutionFailed,
          CONTINUOUS_AGENT_ERROR: values.notifyOnContinuousAgentError,
          DAILY_SUMMARY: values.notifyOnDailySummary,
          WEEKLY_SUMMARY: values.notifyOnWeeklySummary,
          MONTHLY_SUMMARY: values.notifyOnMonthlySummary,
        },
        daily_limit: 0,
      };

      await updateNotificationsMutation.mutateAsync({
        data: notificationPreferences,
      });

      toast({
        title: "Successfully updated notification preferences",
      });
    } catch (error) {
      toast({
        title: "Error updating notifications",
        description:
          error instanceof Error ? error.message : "Something went wrong",
        variant: "destructive",
      });
    }
  }

  function onCancel() {
    form.reset(defaultValues);
  }

  return {
    form,
    onSubmit,
    onCancel,
    isLoading: updateNotificationsMutation.isPending,
  };
}
