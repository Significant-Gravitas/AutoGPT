import { Text } from "@/components/atoms/Text/Text";
import { blockMenuContainerStyle } from "../style";
import { BlockMenuFilters } from "../BlockMenuFilters/BlockMenuFilters";
import { BlockMenuSearchContent } from "../BlockMenuSearchContent/BlockMenuSearchContent";

export const BlockMenuSearch = () => {
  return (
    <div
      className={blockMenuContainerStyle}
      data-id="blocks-control-search-results"
    >
      <BlockMenuFilters />
      <Text variant="body-medium">Search results</Text>
      <BlockMenuSearchContent />
    </div>
  );
};
