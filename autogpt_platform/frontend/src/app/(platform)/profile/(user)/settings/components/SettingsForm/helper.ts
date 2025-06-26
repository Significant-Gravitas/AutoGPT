import { z } from "zod";

export const formSchema = z
  .object({
    email: z.string().email(),
    password: z
      .string()
      .optional()
      .refine((val) => {
        if (val) return val.length >= 12;
        return true;
      }, "String must contain at least 12 character(s)"),
    confirmPassword: z.string().optional(),
    notifyOnAgentRun: z.boolean(),
    notifyOnZeroBalance: z.boolean(),
    notifyOnLowBalance: z.boolean(),
    notifyOnBlockExecutionFailed: z.boolean(),
    notifyOnContinuousAgentError: z.boolean(),
    notifyOnDailySummary: z.boolean(),
    notifyOnWeeklySummary: z.boolean(),
    notifyOnMonthlySummary: z.boolean(),
  })
  .refine((data) => {
    if (data.password || data.confirmPassword) {
      return data.password === data.confirmPassword;
    }
    return true;
  });

export const createDefaultValues = (
  user: { email?: string },
  preferences: { preferences?: Record<string, boolean> },
) => {
  const defaultValues = {
    email: user.email || "",
    password: "",
    confirmPassword: "",
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

  return defaultValues;
};
