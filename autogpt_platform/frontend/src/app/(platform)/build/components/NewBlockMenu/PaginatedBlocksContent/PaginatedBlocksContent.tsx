import { GetV2GetBuilderBlocksParams } from "@/app/api/__generated__/models/getV2GetBuilderBlocksParams";

interface PaginatedBlocksContentProps {
    type: GetV2GetBuilderBlocksParams["type"];
    pageSize?: number;
  }
export const PaginatedBlocksContent = ({ type, pageSize }: PaginatedBlocksContentProps) => {
  return <div>PaginatedBlocksContent</div>;
};