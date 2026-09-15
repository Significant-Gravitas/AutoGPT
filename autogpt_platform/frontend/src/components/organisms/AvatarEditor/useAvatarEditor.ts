import {
  configForName,
  randomConfig,
  type AvatarConfig,
} from "@/components/molecules/BotAvatar/helpers";
import { useState } from "react";
import { findFacet, nextIndex, selectedIndex, type FacetId } from "./helpers";

interface Args {
  value: AvatarConfig;
  onChange: (config: AvatarConfig) => void;
  name?: string;
}

export function useAvatarEditor({ value, onChange, name }: Args) {
  const [facetId, setFacetId] = useState<FacetId>("shape");
  const [seedName, setSeedName] = useState(name ?? "");
  const facet = findFacet(facetId);
  const active = selectedIndex(facet, value);

  function pick(index: number) {
    const option = facet.options[index];
    if (option) onChange(option.config(value));
  }

  function move(step: number) {
    pick(nextIndex(active, facet.options.length, step));
  }

  function surprise() {
    onChange(randomConfig());
  }

  function seedFromName() {
    const trimmed = seedName.trim();
    if (trimmed) onChange(configForName(trimmed));
  }

  return {
    facet,
    facetId,
    setFacetId,
    active,
    pick,
    move,
    seedName,
    setSeedName,
    surprise,
    seedFromName,
  };
}
