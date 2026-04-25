import { z } from "zod";

export const apiKeyConnectSchema = z
  .object({
    title: z
      .string()
      .trim()
      .min(1, "Name is required")
      .max(100, "Name must be 100 characters or less"),
    apiKey: z.string().trim().min(1, "API key is required"),
    expiresAt: z
      .string()
      .trim()
      .optional()
      .refine((value) => !value || !Number.isNaN(Date.parse(value)), {
        message: "Pick a valid date",
      })
      .refine((value) => !value || Date.parse(value) > Date.now(), {
        message: "Expiry must be in the future",
      }),
  })
  .strict();

export type ApiKeyConnectFormValues = z.infer<typeof apiKeyConnectSchema>;
