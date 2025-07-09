import { useState, useEffect } from "react";
import { beautifyString } from "@/lib/utils";
import { Block, BlockUIType } from "@/lib/autogpt-server-api";
import jaro from "jaro-winkler";

// Types for performance optimization
export interface BlockSearchData {
  blockName: string;
  beautifiedName: string;
  description: string;
}

export interface EnhancedBlock extends Block {
  searchData: BlockSearchData;
}

export interface BlockWithAvailability extends Block {
  notAvailable?: string | null;
}

export interface GraphState {
  hasWebhookNodes: boolean;
  hasInputNodes: boolean;
}

// Custom hook for debouncing search input
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

// Memoized function to precompute search data for blocks
export const getBlockSearchData = (
  block: Pick<Block, "name" | "description">,
): BlockSearchData => ({
  blockName: block.name.toLowerCase(),
  beautifiedName: beautifyString(block.name).toLowerCase(),
  description: block.description.toLowerCase(),
});

// Optimized search matching function
export const matchesSearch = (block: EnhancedBlock, query: string): number => {
  if (!query) return 1;

  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery.split(/\s+/);
  const { blockName, beautifiedName, description } = block.searchData;

  // 1. Exact match in name (highest priority)
  if (
    blockName.includes(normalizedQuery) ||
    beautifiedName.includes(normalizedQuery)
  ) {
    return 3;
  }

  // 2. All query words in name (regardless of order)
  const allWordsInName = queryWords.every(
    (word) => blockName.includes(word) || beautifiedName.includes(word),
  );
  if (allWordsInName) return 2;

  // 3. Similarity with name (Jaro-Winkler) - Only for short queries to avoid performance issues
  if (normalizedQuery.length <= 12) {
    const similarityThreshold = 0.65;
    const nameSimilarity = jaro(blockName, normalizedQuery);
    const beautifiedSimilarity = jaro(beautifiedName, normalizedQuery);
    const maxSimilarity = Math.max(nameSimilarity, beautifiedSimilarity);
    if (maxSimilarity > similarityThreshold) {
      return 1 + maxSimilarity; // Score between 1 and 2
    }
  }

  // 4. All query words in description (lower priority)
  const allWordsInDescription = queryWords.every((word) =>
    description.includes(word),
  );
  if (allWordsInDescription) return 0.5;

  return 0;
};

// Helper to check block availability based on graph state
export const getBlockAvailability = (
  block: Block,
  graphState: GraphState,
): string | null => {
  if (block.uiType === BlockUIType.WEBHOOK && graphState.hasWebhookNodes) {
    return "Agents can only have one webhook-triggered block";
  }

  if (block.uiType === BlockUIType.WEBHOOK && graphState.hasInputNodes) {
    return "Webhook-triggered blocks can't be used together with input blocks";
  }

  if (block.uiType === BlockUIType.INPUT && graphState.hasWebhookNodes) {
    return "Input blocks can't be used together with a webhook-triggered block";
  }

  return null;
};

// Helper to extract unique categories from blocks
export const extractCategories = (blocks: Block[]): (string | null)[] => {
  return Array.from(
    new Set([
      null,
      ...blocks
        .flatMap((block) => block.categories.map((cat) => cat.category))
        .sort(),
    ]),
  );
};
