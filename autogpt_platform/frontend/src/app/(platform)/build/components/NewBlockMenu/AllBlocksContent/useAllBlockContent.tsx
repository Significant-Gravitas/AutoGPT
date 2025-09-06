import { getGetV2GetBuilderBlockCategoriesQueryKey, getV2GetBuilderBlocks, useGetV2GetBuilderBlockCategories, useGetV2GetBuilderBlocks } from "@/app/api/__generated__/endpoints/default/default";
import { BlockCategoryResponse } from "@/app/api/__generated__/models/blockCategoryResponse";
import { BlockResponse } from "@/app/api/__generated__/models/blockResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useState } from "react";

export const useAllBlockContent = () => {

    const [loadingCategories, setLoadingCategories] = useState<Set<string>>(new Set());
    const [errorLoadingCategories, setErrorLoadingCategories] = useState<Set<string>>(new Set());
  
    const {data, isLoading, isError, error} = useGetV2GetBuilderBlockCategories(undefined,{
    query: {
      select: (x) => {
        return x.data as BlockCategoryResponse[];
      },
    }
  });


  const handleRefetchBlocks = async (targetCategory: string) => {

    try {
        setLoadingCategories(prev => new Set(prev).add(targetCategory));
        
        // Clear any previous error for this category
        setErrorLoadingCategories(prev => {
            const newSet = new Set(prev);
            newSet.delete(targetCategory);
            return newSet;
        });
        const response = await getV2GetBuilderBlocks({ category: targetCategory });
        await new Promise((resolve) => setTimeout(resolve, 3000));
        const result = response.data as BlockResponse;
        if (result.blocks) {
            const categoriesQueryKey = getGetV2GetBuilderBlockCategoriesQueryKey();
            
            const queryClient = getQueryClient();
            queryClient.setQueryData(categoriesQueryKey, (old: any) => {
                if(!old?.data) return old;
                const categories = old.data as BlockCategoryResponse[];
    
                const updatedCategories = categories.map((old_cat) => {
                    if(old_cat.name === targetCategory) {
                        return {
                            ...old_cat,
                            blocks: result.blocks,
                        }
                    }
                    return old_cat;
                })
                return {
                    ...old,
                    data: updatedCategories
                };
            })
        }
    } catch (error) {
        setErrorLoadingCategories(prev => new Set(prev).add(targetCategory));
    } finally {
        setLoadingCategories(prev => {
            const newSet = new Set(prev);
            newSet.delete(targetCategory);
            return newSet;
        });
    }

  }

  const isLoadingMore = (categoryName: string) => loadingCategories.has(categoryName);
  const isErrorOnLoadingMore = (categoryName: string) => errorLoadingCategories.has(categoryName);


  return {data, isLoading, isError, error, handleRefetchBlocks, isLoadingMore, isErrorOnLoadingMore};
};