import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../helpers";
import { SectionWrap } from "../other/SectionWrap";

interface Props {
  agentName: string;
  agentId: string;
  children: React.ReactNode;
}

export function SelectedViewLayout(props: Props) {
  return (
    <SectionWrap className="relative mb-3 flex min-h-0 flex-1 flex-col">
      <div
        className={`${AGENT_LIBRARY_SECTION_PADDING_X} flex-shrink-0 border-b border-zinc-100 pb-0 lg:pb-4`}
      >
        <Breadcrumbs
          items={[
            { name: "My Library", link: "/library" },
            { name: props.agentName, link: `/library/agents/${props.agentId}` },
          ]}
        />
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-visible">
        {props.children}
      </div>
    </SectionWrap>
  );
}
