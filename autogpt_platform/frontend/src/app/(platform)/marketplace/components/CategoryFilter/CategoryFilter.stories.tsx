import type { Meta, StoryObj } from "@storybook/nextjs";
import { fn } from "storybook/test";
import { useState } from "react";
import { getGetV2ListStoreCategoriesMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import {
  BUILTIN_EXPERT_AVATARS,
  EXPERT_AVATARS,
} from "@/components/molecules/ExpertAvatar/helpers";
import { ExpertCard } from "../ExpertsSection/components/ExpertCard";
import { CategoryFilter } from "./CategoryFilter";

const meta = {
  title: "Marketplace/CategoryColors",
  component: CategoryFilter,
  args: { selected: "finance", onSelect: fn() },
  parameters: {
    msw: {
      handlers: [
        getGetV2ListStoreCategoriesMockHandler(
          EXPERT_AVATARS.map((avatar) => ({
            value: avatar.id,
            label: avatar.id[0].toUpperCase() + avatar.id.slice(1),
            description: avatar.label,
          })),
        ),
      ],
    },
  },
} satisfies Meta<typeof CategoryFilter>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Experts: Story = {
  render: function Render() {
    return <CategoryGallery />;
  },
};

function CategoryGallery() {
  const [category, setCategory] = useState<string | null>("finance");
  const avatars = BUILTIN_EXPERT_AVATARS.filter(
    (avatar) => !category || Object.keys(avatar.variants).includes(category),
  );
  return (
    <div className="w-full max-w-6xl p-4">
      <CategoryFilter selected={category} onSelect={setCategory} />
      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {avatars.map((avatar) => (
          <ExpertCard
            key={avatar.id}
            expert={exampleExpert(avatar)}
            category={category}
            isHired={false}
          />
        ))}
      </div>
    </div>
  );
}

function exampleExpert(
  avatar: (typeof BUILTIN_EXPERT_AVATARS)[number],
): ExpertTemplate {
  return {
    id: avatar.id,
    name: avatar.name,
    avatar_url: avatar.url,
    role: avatar.primary_category,
    categories: Object.keys(avatar.variants),
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
