import { z } from "zod";

export const apiKeyConnectSchema = z.object({
  title: z
    .string()
    .trim()
    .min(1, "Name is required")
    .max(100, "Name must be 100 characters or less"),
  apiKey: z.string().trim().min(1, "API key is required"),
});

export type ApiKeyConnectFormValues = z.infer<typeof apiKeyConnectSchema>;
