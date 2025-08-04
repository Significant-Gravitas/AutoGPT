import { Separator } from "@/components/ui/separator";
import { LibraryActionHeader2 } from "../LibraryActionHeader2/LibraryActionHeader2";
import { LibraryAgentList2 } from "../LibraryAgentList2/LibraryAgentList2";
import { LibraryActionSubHeader2 } from "../LibraryActionSubHeader2/LibraryActionSubHeader2";

export const LibraryMainView = () => {
  return (
    <div>
      <LibraryActionHeader2 />
      <LibraryActionSubHeader2 />
      <Separator className="mt-2.5 mb-4"/>
      <LibraryAgentList2 />
    </div>
  );
};
