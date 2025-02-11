import { cn } from '@/lib/utils';
import { Check } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

type OnboardingListElementProps = {
  label: string;
  text: string;
  selected?: boolean;
  custom?: boolean;
  onClick: (content: string) => void;
};

export function OnboardingListElement({ label, text, selected, custom, onClick }: OnboardingListElementProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [content, setContent] = useState('');

  useEffect(() => {
    if (selected && custom && inputRef.current) {
      inputRef.current.focus();
    }
  }, [selected, custom]);

  const setCustomText = (e: React.ChangeEvent<HTMLInputElement>) => {
    setContent(e.target.value);
    onClick(e.target.value);
  }

  return (
    <button
      onClick={() => onClick(content)}
      className={`
        relative w-[530px] h-[78px] px-5 py-4 bg-white rounded-xl
        flex items-center
        transition-all duration-100
        ${selected ?
          'border-2 border-violet-700 bg-[#F5F3FF80]' :
          'border-2 border-transparent hover:border hover:border-zinc-400'
        }
      `}
    >
      <div className="flex flex-col items-start gap-1">
        <span className="text-zinc-700 text-sm font-medium">{label}</span>
        {custom && selected ?
          <input
            ref={inputRef}
            className={cn(selected ? "text-zinc-600" : "text-zinc-400", "text-sm font-poppins border-0 bg-[#F5F3FF80] focus:outline-none")}
            placeholder={text}
            value={content}
            onChange={setCustomText} /> :
          <span className={cn(selected ? "text-zinc-600" : "text-zinc-400", "text-sm")}>{text}</span>}
      </div>
      {selected && !custom && (
        <div className="absolute right-4">
          <Check size={24} className="text-violet-700" />
        </div>
      )}
    </button>
  );
};

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

const OnboardingList = ({ className, elements, selectedId, onSelect }: OnboardingListProps) => {
  const [isCustom, setIsCustom] = useState(false);

  useEffect(() => {
    // If selectedId is not set, set isCustom to false
    if (selectedId === undefined) {
      setIsCustom(false);
      return;
    }
    // Search elements for selectedId, if not found, set isCustom to true
    const found = elements.find((element) => element.id === selectedId);
    setIsCustom(!found);
  }, [selectedId]);

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
        text="Please specify"
        selected={isCustom}
        custom
        onClick={onSelect}
      />
    </div>
  );
};

export default OnboardingList;
