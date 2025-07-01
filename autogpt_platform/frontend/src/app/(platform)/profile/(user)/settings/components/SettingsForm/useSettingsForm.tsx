"use client";
import { useForm } from "react-hook-form";
import { createDefaultValues, formSchema } from "./helper";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { updateSettings } from "../../actions";
import { useToast } from "@/components/ui/use-toast";
import { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import { User } from "@supabase/supabase-js";

export const useSettingsForm = ({
  preferences,
  user,
}: {
  preferences: NotificationPreference;
  user: User;
}) => {
  const { toast } = useToast();
  const defaultValues = createDefaultValues(user, preferences);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues,
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    try {
      const formData = new FormData();

      Object.entries(values).forEach(([key, value]) => {
        if (key !== "confirmPassword") {
          formData.append(key, value.toString());
        }
      });

      await updateSettings(formData);

      toast({
        title: "Successfully updated settings",
      });
    } catch (error) {
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Something went wrong",
        variant: "destructive",
      });
      throw error;
    }
  }

  function onCancel() {
    form.reset(defaultValues);
  }

  return { form, onSubmit, onCancel };
};
