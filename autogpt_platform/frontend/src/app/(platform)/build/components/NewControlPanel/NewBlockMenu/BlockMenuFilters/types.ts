import type { SearchEntry } from "@/app/api/__generated__/models/searchEntry";

export type CategoryKey = NonNullable<SearchEntry["filter"]>[number];

export type CategoryCounts = Record<CategoryKey, number>;
