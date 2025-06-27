"use client";
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import React from "react";

const TestingComponent = () => {
  const {
    data: libraryListData,
    isLoading,
    isError,
  } = useGetV2ListLibraryAgents(undefined, {
    query: {
      select: (x) => {
        return x.data as LibraryAgentResponse;
      },
    },
  });

  if (isLoading) {
    return "Loading...";
  }

  if (isError) {
    return "Error...";
  }

  return <div>{libraryListData && JSON.stringify(libraryListData)}</div>;
};

export default TestingComponent;
