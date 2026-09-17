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
      .superRefine((value, ctx) => {
        if (!value) return;
        const parsed = Date.parse(value);
        if (Number.isNaN(parsed)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Pick a valid date",
          });
          return;
        }
        if (parsed <= Date.now()) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Expiry must be in the future",
          });
        }
      }),
  })
  .strict();

export type ApiKeyConnectFormValues = z.infer<typeof apiKeyConnectSchema>;
