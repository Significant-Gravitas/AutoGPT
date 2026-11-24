"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import dynamic from "next/dynamic";

// The form library, its validator, the date picker and the custom fields load
// when a form first shows, so a page that may show one doesn't carry them.
export const LazyFormRenderer = dynamic(
  () => import("./FormRenderer").then((module) => module.FormRenderer),
  { ssr: false, loading: () => <Skeleton className="mt-4 h-10 w-full" /> },
);
