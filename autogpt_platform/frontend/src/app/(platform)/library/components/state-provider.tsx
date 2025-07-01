"use client";

import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import {
  createContext,
  useState,
  ReactNode,
  useContext,
  Dispatch,
  SetStateAction,
} from "react";

interface LibraryPageContextType {
  searchTerm: string;
  setSearchTerm: Dispatch<SetStateAction<string>>;
  uploadedFile: File | null;
  setUploadedFile: Dispatch<SetStateAction<File | null>>;
  librarySort: LibraryAgentSort;
  setLibrarySort: Dispatch<SetStateAction<LibraryAgentSort>>;
}

export const LibraryPageContext = createContext<LibraryPageContextType>(
  {} as LibraryPageContextType,
);

export function LibraryPageStateProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [librarySort, setLibrarySort] = useState<LibraryAgentSort>(
    LibraryAgentSort.updatedAt,
  );

  return (
    <LibraryPageContext.Provider
      value={{
        searchTerm,
        setSearchTerm,
        uploadedFile,
        setUploadedFile,
        librarySort,
        setLibrarySort,
      }}
    >
      {children}
    </LibraryPageContext.Provider>
  );
}

export function useLibraryPageContext(): LibraryPageContextType {
  const context = useContext(LibraryPageContext);
  if (!context) {
    throw new Error("Error in context of Library page");
  }
  return context;
}
