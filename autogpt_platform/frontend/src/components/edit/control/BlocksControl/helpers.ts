import { beautifyString } from "@/lib/utils";
import { Block, BlockUIType } from "@/lib/autogpt-server-api";
import jaro from "jaro-winkler";

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

export function getBlockSearchData(
  block: Pick<Block, "name" | "description">,
): BlockSearchData {
  return {
    blockName: block.name.toLowerCase(),
    beautifiedName: beautifyString(block.name).toLowerCase(),
    description: block.description.toLowerCase(),
  };
}

export function matchesSearch(block: EnhancedBlock, query: string): number {
  if (!query) return 1;

  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery.split(/\s+/);
  const { blockName, beautifiedName, description } = block.searchData;

  // Exact match in name (highest priority)
  if (
    blockName.includes(normalizedQuery) ||
    beautifiedName.includes(normalizedQuery)
  ) {
    return 3;
  }

  // All query words in name (regardless of order)
  const allWordsInName = queryWords.every(
    (word) => blockName.includes(word) || beautifiedName.includes(word),
  );
  if (allWordsInName) return 2;

  // Similarity with name (Jaro-Winkler) - Only for short queries to avoid performance issues
  if (normalizedQuery.length <= 12) {
    const similarityThreshold = 0.65;
    const nameSimilarity = jaro(blockName, normalizedQuery);
    const beautifiedSimilarity = jaro(beautifiedName, normalizedQuery);
    const maxSimilarity = Math.max(nameSimilarity, beautifiedSimilarity);
    if (maxSimilarity > similarityThreshold) {
      return 1 + maxSimilarity; // Score between 1 and 2
    }
  }

  // All query words in description (lower priority)
  const allWordsInDescription = queryWords.every((word) =>
    description.includes(word),
  );
  if (allWordsInDescription) return 0.5;

  return 0;
}

export function getBlockAvailability(
  block: Block,
  graphState: GraphState,
): string | null {
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
}

export function extractCategories(blocks: Block[]): (string | null)[] {
  return Array.from(
    new Set([
      null,
      ...blocks
        .flatMap((block) => block.categories.map((cat) => cat.category))
        .sort(),
    ]),
  );
}
