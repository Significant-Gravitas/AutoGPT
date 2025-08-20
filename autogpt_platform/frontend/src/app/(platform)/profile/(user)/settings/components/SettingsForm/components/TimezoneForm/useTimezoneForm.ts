"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { User } from "@supabase/supabase-js";
import {
  usePostV1UpdateUserTimezone,
  getGetV1GetUserTimezoneQueryKey,
} from "@/app/api/__generated__/endpoints/auth/auth";
import { useQueryClient } from "@tanstack/react-query";

const formSchema = z.object({
  timezone: z.string().min(1, "Please select a timezone"),
});

type FormData = z.infer<typeof formSchema>;

type UseTimezoneFormProps = {
  user: User;
  currentTimezone: string;
};

export const useTimezoneForm = ({ currentTimezone }: UseTimezoneFormProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const form = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      timezone: currentTimezone,
    },
  });

  const updateTimezone = usePostV1UpdateUserTimezone();

  const onSubmit = async (data: FormData) => {
    setIsLoading(true);
    try {
      await updateTimezone.mutateAsync({
        data: { timezone: data.timezone } as any,
      });

      // Invalidate the timezone query to refetch the updated value
      await queryClient.invalidateQueries({
        queryKey: getGetV1GetUserTimezoneQueryKey(),
      });

      toast({
        title: "Success",
        description: "Your timezone has been updated successfully.",
        variant: "success",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to update timezone. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return {
    form,
    onSubmit,
    isLoading,
  };
};
