import z from "zod";

export const publishAgentSchema = z.object({
  title: z
    .string()
    .min(1, "Title is required")
    .max(100, "Title must be less than 100 characters"),
  subheader: z
    .string()
    .min(1, "Subheader is required")
    .max(200, "Subheader must be less than 200 characters"),
  slug: z
    .string()
    .min(1, "Slug is required")
    .max(50, "Slug must be less than 50 characters")
    .regex(
      /^[a-z0-9-]+$/,
      "Slug can only contain lowercase letters, numbers, and hyphens",
    ),
  youtubeLink: z
    .string()
    .optional()
    .refine((val) => {
      if (!val) return true;
      try {
        const url = new URL(val);
        const allowedHosts = [
          "youtube.com",
          "www.youtube.com",
          "youtu.be",
          "www.youtu.be",
        ];
        return allowedHosts.includes(url.hostname);
      } catch {
        return false;
      }
    }, "Please enter a valid YouTube URL"),
  category: z.string().min(1, "Category is required"),
  description: z
    .string()
    .min(1, "Description is required")
    .max(1000, "Description must be less than 1000 characters"),
  recommendedScheduleCron: z.string().optional(),
});

export type PublishAgentFormData = z.infer<typeof publishAgentSchema>;

export interface PublishAgentInfoInitialData {
  agent_id: string;
  title: string;
  subheader: string;
  slug: string;
  thumbnailSrc: string;
  youtubeLink: string;
  category: string;
  description: string;
  additionalImages?: string[];
  recommendedScheduleCron?: string;
}
