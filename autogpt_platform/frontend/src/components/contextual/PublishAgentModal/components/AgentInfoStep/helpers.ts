import z from "zod";
import { validateYouTubeUrl } from "@/lib/utils";

// Create conditional schema that changes based on whether it's a marketplace update
export const publishAgentSchemaFactory = (
  isMarketplaceUpdate: boolean = false,
) => {
  const baseSchema = {
    changesSummary: isMarketplaceUpdate
      ? z
          .string()
          .min(1, "Changes summary is required for updates")
          .max(500, "Changes summary must be less than 500 characters")
      : z.string().optional(),
    title: isMarketplaceUpdate
      ? z
          .string()
          .optional()
          .refine(
            (val) => !val || val.length <= 100,
            "Title must be less than 100 characters",
          )
      : z
          .string()
          .min(1, "Title is required")
          .max(100, "Title must be less than 100 characters"),
    subheader: isMarketplaceUpdate
      ? z
          .string()
          .optional()
          .refine(
            (val) => !val || val.length <= 200,
            "Subheader must be less than 200 characters",
          )
      : z
          .string()
          .min(1, "Subheader is required")
          .max(200, "Subheader must be less than 200 characters"),
    slug: isMarketplaceUpdate
      ? z
          .string()
          .optional()
          .refine(
            (val) => !val || (val.length <= 50 && /^[a-z0-9-]+$/.test(val)),
            "Slug can only contain lowercase letters, numbers, and hyphens",
          )
      : z
          .string()
          .min(1, "Slug is required")
          .max(50, "Slug must be less than 50 characters")
          .regex(
            /^[a-z0-9-]+$/,
            "Slug can only contain lowercase letters, numbers, and hyphens",
          ),
    youtubeLink: isMarketplaceUpdate
      ? z
          .string()
          .optional()
          .refine(
            (val) => !val || validateYouTubeUrl(val),
            "Please enter a valid YouTube URL",
          )
      : z
          .string()
          .refine(validateYouTubeUrl, "Please enter a valid YouTube URL"),
    category: isMarketplaceUpdate
      ? z.string().optional()
      : z.string().min(1, "Category is required"),
    description: isMarketplaceUpdate
      ? z
          .string()
          .optional()
          .refine(
            (val) => !val || val.length <= 1000,
            "Description must be less than 1000 characters",
          )
      : z
          .string()
          .min(1, "Description is required")
          .max(1000, "Description must be less than 1000 characters"),
    recommendedScheduleCron: z.string().optional(),
    instructions: z
      .string()
      .optional()
      .refine(
        (val) => !val || val.length <= 2000,
        "Instructions must be less than 2000 characters",
      ),
    agentOutputDemo: isMarketplaceUpdate
      ? z
          .string()
          .optional()
          .refine(
            (val) => !val || validateYouTubeUrl(val),
            "Please enter a valid YouTube URL",
          )
      : z
          .string()
          .refine(validateYouTubeUrl, "Please enter a valid YouTube URL"),
  };

  return z.object(baseSchema);
};

// Default schema for backwards compatibility
export const publishAgentSchema = publishAgentSchemaFactory(false);

export type PublishAgentFormData = z.infer<
  ReturnType<typeof publishAgentSchemaFactory>
>;

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
  instructions?: string;
  agentOutputDemo?: string;
  changesSummary?: string;
}
