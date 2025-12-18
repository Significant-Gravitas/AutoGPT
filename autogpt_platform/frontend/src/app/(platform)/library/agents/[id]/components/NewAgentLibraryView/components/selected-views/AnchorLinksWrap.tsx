import { cn } from "@/lib/utils";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../helpers";

type Props = {
  children: React.ReactNode;
};

export function AnchorLinksWrap({ children }: Props) {
  return (
    <div className={cn(AGENT_LIBRARY_SECTION_PADDING_X, "hidden lg:block")}>
      <nav className="flex gap-8 px-3 pb-1">{children}</nav>
    </div>
  );
}
