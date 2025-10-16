import { useRef } from "react";

interface UseGraphMenuSearchBarComponentProps {
  onSearchChange: (query: string) => void;
}

export const useGraphMenuSearchBarComponent = ({
  onSearchChange,
}: UseGraphMenuSearchBarComponentProps) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClear = () => {
    onSearchChange("");
    inputRef.current?.focus();
  };

  return {
    inputRef,
    handleClear,
  };
};