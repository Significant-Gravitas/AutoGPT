"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import type { User } from "@/lib/auth/types";
import { usePostV1UpdateNotificationPreferences } from "@/app/api/__generated__/endpoints/auth/auth";

// The volume knob from the Briefing footer, not a checkbox list. Billing and
// account messages are service mail and are deliberately absent — they are
// sent regardless of what is set here.
const notificationFormSchema = z.object({
  briefingFrequency: z.enum(["DAILY", "WEEKLY", "MONTHLY", "OFF"]),
  alertsEnabled: z.boolean(),
  storeVerdictsEnabled: z.boolean(),
});

export type NotificationFormValues = z.infer<typeof notificationFormSchema>;

function createNotificationDefaultValues(
  preferences: NotificationPreference,
): NotificationFormValues {
  return {
    briefingFrequency: (preferences.briefing_frequency ??
      "WEEKLY") as NotificationFormValues["briefingFrequency"],
    alertsEnabled: preferences.alerts_enabled ?? true,
    storeVerdictsEnabled: preferences.store_verdicts_enabled ?? true,
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

  const form = useForm<NotificationFormValues>({
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

  async function onSubmit(values: NotificationFormValues) {
    try {
      await updateNotificationsMutation.mutateAsync({
        data: {
          email: user.email || "",
          briefing_frequency: values.briefingFrequency,
          alerts_enabled: values.alertsEnabled,
          store_verdicts_enabled: values.storeVerdictsEnabled,
        },
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
