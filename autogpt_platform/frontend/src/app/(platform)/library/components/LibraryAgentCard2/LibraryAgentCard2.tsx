import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/ui/separator";
import Image from "next/image";

export const LibraryAgentCard2 = () => {
  return (
    <div className="rounded-medium h-44 bg-white p-2 pl-3 space-y-2">
      {/* Destination */}
      <div className="flex items-center gap-2">
        <span className="w-3 h-3 rounded-full bg-green-400" />
        <Text variant="small-medium" className="!text-zinc-400 uppercase !leading-5 tracking-[0.1em]">From Marketplace</Text>
      </div>

      {/* Information */}
      <div className="pb-1 flex  gap-4 justify-between">
        <div className="flex justify-between flex-col">
          <Text variant="large-medium">AI Text Generator</Text>

          <div className="flex flex-row gap-2">
            <Text variant="small-medium">1 min ago</Text>
            <Text variant="small-medium">12 runs</Text>
          </div>
        </div>
        <div className="h-[4.75rem] aspect-video relative overflow-hidden rounded-small">
            <Image src="/placeholder.png" alt="Agent-image" fill className="object-cover" />
        </div>
      </div>

      <Separator className="border-zinc-200"/>


      {/* Actions */}
      <div>

      </div>
    </div>
  );
};
