import { useState, useMemo } from "react";
import debounce from "lodash/debounce";
import {
  Block,
  BlockUIType,
  SpecialBlockID,
  GraphMeta,
} from "@/lib/autogpt-server-api";
import { CustomNode } from "@/components/CustomNode";
import {
  getBlockSearchData,
  matchesSearch,
  getBlockAvailability,
  extractCategories,
  EnhancedBlock,
  BlockWithAvailability,
  GraphState,
} from "./helpers";

interface Args {
  blocks: Block[];
  flows: GraphMeta[];
  nodes: CustomNode[];
  addBlock: (
    id: string,
    name: string,
    hardcodedValues: Record<string, any>,
  ) => void;
}

export function useBlocksControl({ blocks, flows, nodes, addBlock }: Args) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // Memoize graph state checks to avoid recalculating on every render
  const graphState = useMemo(
    (): GraphState => ({
      hasWebhookNodes: nodes.some((n) =>
        [BlockUIType.WEBHOOK, BlockUIType.WEBHOOK_MANUAL].includes(
          n.data.uiType,
        ),
      ),
      hasInputNodes: nodes.some((n) => n.data.uiType === BlockUIType.INPUT),
    }),
    [nodes],
  );

  // Memoize blocks with precomputed search data
  const blocksWithSearchData = useMemo((): EnhancedBlock[] => {
    return blocks.map((block) => ({
      ...block,
      searchData: getBlockSearchData(block),
    }));
  }, [blocks]);

  // Memoize agent blocks list with search data
  const agentBlocksWithSearchData = useMemo((): EnhancedBlock[] => {
    return flows.map((flow) => {
      const description = `Ver.${flow.version}${flow.description ? ` | ${flow.description}` : ""}`;
      return {
        id: SpecialBlockID.AGENT,
        name: flow.name,
        description,
        categories: [{ category: "AGENT", description: "" }],
        inputSchema: flow.input_schema,
        outputSchema: flow.output_schema,
        staticOutput: false,
        uiType: BlockUIType.AGENT,
        uiKey: flow.id,
        costs: [],
        hardcodedValues: {
          graph_id: flow.id,
          graph_version: flow.version,
          input_schema: flow.input_schema,
          output_schema: flow.output_schema,
        },
        searchData: getBlockSearchData({ name: flow.name, description }),
      } satisfies EnhancedBlock;
    });
  }, [flows]);

  // Memoize filtered and sorted blocks
  const filteredAvailableBlocks = useMemo((): BlockWithAvailability[] => {
    const blockList = blocksWithSearchData
      .filter((b) => b.uiType !== BlockUIType.AGENT)
      .sort((a, b) => a.name.localeCompare(b.name));

    const allBlocks = blockList.concat(agentBlocksWithSearchData);

    return allBlocks
      .map((block) => ({
        block,
        score: matchesSearch(block, searchQuery),
      }))
      .filter(
        ({ block, score }) =>
          score > 0 &&
          (!selectedCategory ||
            block.categories.some((cat) => cat.category === selectedCategory)),
      )
      .sort((a, b) => b.score - a.score)
      .map(({ block }) => ({
        ...block,
        notAvailable: getBlockAvailability(block, graphState),
      }));
  }, [
    blocksWithSearchData,
    agentBlocksWithSearchData,
    searchQuery,
    selectedCategory,
    graphState,
  ]);

  const categories = useMemo(() => extractCategories(blocks), [blocks]);

  // Create debounced version of setSearchQuery
  const debouncedSetSearchQuery = useMemo(
    () => debounce(setSearchQuery, 200),
    [],
  );

  function resetFilters() {
    setSearchQuery("");
    setSelectedCategory(null);
  }

  function handleCategoryClick(category: string | null) {
    setSelectedCategory(selectedCategory === category ? null : category);
  }

  function handleAddBlock(block: BlockWithAvailability) {
    if (!block.notAvailable) {
      addBlock(block.id, block.name, block?.hardcodedValues || {});
    }
  }

  return {
    searchQuery,
    setSearchQuery: debouncedSetSearchQuery,
    selectedCategory,
    filteredAvailableBlocks,
    categories,
    resetFilters,
    handleCategoryClick,
    handleAddBlock,
  };
}
