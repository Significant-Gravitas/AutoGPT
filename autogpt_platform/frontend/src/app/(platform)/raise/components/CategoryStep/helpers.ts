import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { EXPERT_AVATARS } from "@/components/molecules/ExpertAvatar/helpers";

export interface CategoryOption {
  id: ExpertAvatarRequestCategory;
  label: string;
  hex: string;
}

export const CATEGORY_OPTIONS: CategoryOption[] = Object.values(
  ExpertAvatarRequestCategory,
).map((id) => ({
  id,
  label: id[0].toUpperCase() + id.slice(1),
  hex: EXPERT_AVATARS.find((avatar) => avatar.id === id)?.hex ?? "#B5ADA0",
}));

export function categoryOptionsForSelection(
  selected: ExpertAvatarRequestCategory | null,
) {
  if (!selected) return CATEGORY_OPTIONS;
  return CATEGORY_OPTIONS.filter((option) => option.id === selected);
}

/** The wizard color a category answers for, used from the avatar beat onwards. */
export function colorForCategory(category: ExpertAvatarRequestCategory) {
  return (
    EXPERT_AVATARS.find((avatar) => avatar.id === category)?.color ??
    "amber-300"
  );
}
