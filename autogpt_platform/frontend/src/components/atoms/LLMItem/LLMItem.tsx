import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";
import claudeImg from "./assets/claude.svg";
import gptImg from "./assets/gpt.svg";
import perplexityImg from "./assets/perplexity.svg";

type LLMType = "claude" | "gpt" | "perplexity";

const llmTypeMap: Record<LLMType, { image: string; name: string }> = {
  claude: {
    image: claudeImg.src,
    name: "Claude",
  },
  gpt: {
    image: gptImg.src,
    name: "GPT",
  },
  perplexity: {
    image: perplexityImg.src,
    name: "Perplexity",
  },
};

type Props = {
  type: LLMType;
};

export function LLMItem({ type }: Props) {
  return (
    <div className="flex flex-nowrap items-center gap-2">
      <Image
        src={llmTypeMap[type].image}
        alt={llmTypeMap[type].name}
        width={40}
        height={40}
        className="h-5 w-5 rounded-xsmall"
      />
      <Text variant="body">{llmTypeMap[type].name}</Text>
    </div>
  );
}
