import React, { useState, useEffect, Fragment } from "react";
import Block from "../Block";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { allBlocksDataWithCategories } from "../../testing_data";
import { Skeleton } from "@/components/ui/skeleton";

type BlockItem = {
  title: string;
  description: string;
};

export type BlockCategory = {
  name: string;
  count: number;
  items: BlockItem[];
};

const AllBlocksContent: React.FC = () => {
  const [categories, setCategories] = useState<BlockCategory[]>([]);
  const [loading, setLoading] = useState(true);

  // TEMPORARY FETCHING
  useEffect(() => {
    const fetchBlocks = async () => {
      setLoading(true);
      setTimeout(() => {
        setCategories(allBlocksDataWithCategories);
        setLoading(false);
      }, 800);
    };

    fetchBlocks();
  }, []);

  if (loading) {
    return (
      <div className="w-full space-y-3 p-4">
        {[0, 1, 3].map((categoryIndex) => (
          <Fragment key={categoryIndex}>
            {categoryIndex > 0 && (
              <Skeleton className="my-4 h-[1px] w-full text-zinc-100" />
            )}
            {[0, 1, 2].map((blockIndex) => (
              <Block.Skeleton key={`${categoryIndex}-${blockIndex}`} />
            ))}
          </Fragment>
        ))}
      </div>
    );
  }

  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <div className="w-full space-y-3 px-4 pb-4">
        {categories.map((category, index) => (
          <Fragment key={category.name}>
            {index > 0 && (
              <Separator className="h-[1px] w-full text-zinc-300" />
            )}

            {/* Category Section */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
                  {category.name}
                </p>
                <span className="rounded-full bg-zinc-100 px-[0.375rem] font-sans text-sm leading-[1.375rem] text-zinc-600">
                  {category.count}
                </span>
              </div>

              <div className="space-y-2">
                {category.items.slice(0, 3).map((item, idx) => (
                  <Block
                    key={`${category.name}-${idx}`}
                    title={item.title}
                    description={item.description}
                  />
                ))}

                {category.items.length > 3 && (
                  <Button
                    variant={"link"}
                    className="px-0 font-sans text-sm leading-[1.375rem] text-zinc-600 underline hover:text-zinc-800"
                  >
                    see all
                  </Button>
                )}
              </div>
            </div>
          </Fragment>
        ))}
      </div>
    </div>
  );
};

export default AllBlocksContent;
