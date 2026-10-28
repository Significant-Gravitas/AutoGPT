import type { Meta, StoryObj } from "@storybook/nextjs";
import { fn } from "storybook/test";
import { useState } from "react";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getGetV2ListStoreCategoriesMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import {
  EXPERT_PALETTE,
  MANAGED_IDENTITIES,
} from "@/components/molecules/ExpertAvatar/helpers";
import { ExpertCard } from "../ExpertsSection/components/ExpertCard";
import { CategoryFilter } from "./CategoryFilter";

const CATEGORIES = [
  "marketing",
  "sales",
  "finance",
  "support",
  "operations",
  "research",
  "content",
  "development",
] as const;

const meta = {
  title: "Marketplace/CategoryColors",
  component: CategoryFilter,
  decorators: [
    (Story) => (
      <BackendAPIProvider>
        <Story />
      </BackendAPIProvider>
    ),
  ],
  args: { selected: "finance", onSelect: fn() },
  parameters: {
    msw: {
      handlers: [
        getGetV2ListStoreCategoriesMockHandler(
          CATEGORIES.map((value) => ({
            value,
            label: value[0].toUpperCase() + value.slice(1),
            description: EXPERT_PALETTE[value].label,
          })),
        ),
      ],
    },
  },
} satisfies Meta<typeof CategoryFilter>;
export default meta;
type Story = StoryObj<typeof meta>;

/** The expert shelf under each category filter. Filtering changes which
 *  cards show and which tag they wear; every card keeps its own artwork and
 *  its own family color. */
export const Experts: Story = {
  render: function Render() {
    return <CategoryGallery />;
  },
};

function CategoryGallery() {
  const [category, setCategory] = useState<string | null>(null);
  const experts = MANAGED_IDENTITIES.filter(
    (identity) =>
      identity.categories.length > 0 &&
      (!category || identity.categories.includes(category)),
  );
  return (
    <div className="w-full max-w-6xl p-4">
      <CategoryFilter selected={category} onSelect={setCategory} />
      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {experts.map((identity) => (
          <ExpertCard
            key={identity.id}
            expert={exampleExpert(identity)}
            category={category}
            isHired={false}
          />
        ))}
      </div>
    </div>
  );
}

function exampleExpert(
  identity: (typeof MANAGED_IDENTITIES)[number],
): ExpertTemplate {
  return {
    id: identity.id,
    name: identity.name,
    avatar_url: identity.url,
    role: identity.visual_category,
    job_title: identity.job_title,
    categories: identity.categories,
    tagline: null,
    bio: null,
    skills: [],
    identity: "",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: true,
    source_template_id: null,
    is_archived: false,
    workflows: [],
  };
}
