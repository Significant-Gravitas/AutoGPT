import { cn } from "@/lib/utils";
import { Check } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

type OnboardingListElementProps = {
  label: string;
  text: string;
  selected?: boolean;
  custom?: boolean;
  onClick: (content: string) => void;
};

export function OnboardingListElement({
  label,
  text,
  selected,
  custom,
  onClick,
}: OnboardingListElementProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [content, setContent] = useState(text);

  useEffect(() => {
    if (selected && custom && inputRef.current) {
      inputRef.current.focus();
    }
  }, [selected, custom]);

  const setCustomText = (e: React.ChangeEvent<HTMLInputElement>) => {
    setContent(e.target.value);
    onClick(e.target.value);
  };

  return (
    <button
      onClick={() => onClick(content)}
      className={`relative flex h-[78px] w-[530px] items-center rounded-xl px-5 py-4 transition-all duration-200 ${
        selected
          ? "border-2 border-violet-700 bg-[#F5F3FF80]"
          : "border border-transparent bg-white hover:border-zinc-400"
      } `}
    >
      <div className="flex flex-col items-start gap-1">
        <span className="text-sm font-medium text-zinc-700">{label}</span>
        {custom && selected ? (
          <input
            ref={inputRef}
            className={cn(
              selected ? "text-zinc-600" : "text-zinc-400",
              "font-poppins border-0 bg-[#F5F3FF80] text-sm focus:outline-none",
            )}
            placeholder="Please specify"
            value={content}
            onChange={setCustomText}
          />
        ) : (
          <span
            className={cn(
              selected ? "text-zinc-600" : "text-zinc-400",
              "text-sm",
            )}
          >
            {custom ? "Please specify" : text}
          </span>
        )}
      </div>
      {!custom && (
        <div className="absolute right-4">
          <Check
            size={24}
            className={`${selected ? "text-violet-700" : "text-transparent"} transition-all duration-200`}
          />
        </div>
      )}
    </button>
  );
}

type OnboardingListProps = {
  className?: string;
  elements: Array<{
    label: string;
    text: string;
    id: string;
  }>;
  selectedId?: string;
  onSelect: (id: string) => void;
};

const OnboardingList = ({
  className,
  elements,
  selectedId,
  onSelect,
}: OnboardingListProps) => {
  const isCustom = useCallback(() => {
    return (
      selectedId !== undefined &&
      !elements.some((element) => element.id === selectedId)
    );
  }, [selectedId, elements]);

  return (
    <div className={cn(className, "flex flex-col gap-2")}>
      {elements.map((element) => (
        <OnboardingListElement
          key={element.id}
          label={element.label}
          text={element.text}
          selected={element.id === selectedId}
          onClick={() => onSelect(element.id)}
        />
      ))}
      <OnboardingListElement
        label="Other"
        text={isCustom() ? selectedId! : ""}
        selected={isCustom()}
        custom
        onClick={(c) => {
          onSelect(c);
        }}
      />
    </div>
  );
};

export default OnboardingList;
