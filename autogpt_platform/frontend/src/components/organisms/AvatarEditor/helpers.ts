import {
  ACCESSORIES,
  COLORS,
  SHAPES,
  type AvatarConfig,
} from "@/components/molecules/BotAvatar/helpers";

export type FacetId = "shape" | "color" | "accessory";

export interface FacetOption {
  id: string;
  label: string;
  hint: string;
  config: (base: AvatarConfig) => AvatarConfig;
}

export interface Facet {
  id: FacetId;
  label: string;
  options: FacetOption[];
}

export const FACETS: Facet[] = [
  {
    id: "shape",
    label: "Shape",
    options: SHAPES.map((shape) => ({
      id: shape.id,
      label: shape.label,
      hint: shape.hint,
      config: (base) => ({ ...base, shape: shape.id }),
    })),
  },
  {
    id: "color",
    label: "Colour",
    options: COLORS.map((color) => ({
      id: color.id,
      label: color.label,
      hint: color.role,
      config: (base) => ({ ...base, color: color.id }),
    })),
  },
  {
    id: "accessory",
    label: "Accessory",
    options: ACCESSORIES.map((accessory) => ({
      id: accessory.id,
      label: accessory.label,
      hint: accessory.hint,
      config: (base) => ({ ...base, accessory: accessory.id }),
    })),
  },
];

export function findFacet(id: FacetId) {
  return FACETS.find((facet) => facet.id === id) ?? FACETS[0];
}

export function selectedIndex(facet: Facet, config: AvatarConfig) {
  const current = config[facet.id];
  const index = facet.options.findIndex((option) => option.id === current);
  return index < 0 ? 0 : index;
}

// Arrow keys walk a radio group as one wrapping ring, which matches how the
// grid reads regardless of how many columns the viewport happens to give it.
export function nextIndex(
  current: number,
  length: number,
  step: number,
): number {
  return (current + step + length) % length;
}
