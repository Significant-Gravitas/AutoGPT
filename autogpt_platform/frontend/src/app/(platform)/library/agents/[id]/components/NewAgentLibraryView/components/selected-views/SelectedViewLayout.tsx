import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../helpers";
import { AgentSettingsModal } from "../modals/AgentSettingsModal/AgentSettingsModal";
import { SectionWrap } from "../other/SectionWrap";

interface Props {
  agent: LibraryAgent;
  children: React.ReactNode;
  banner?: React.ReactNode;
  additionalBreadcrumb?: { name: string; link?: string };
}

export function SelectedViewLayout(props: Props) {
  return (
    <SectionWrap className="relative mb-3 flex min-h-0 flex-1 flex-col">
      <div
        className={`${AGENT_LIBRARY_SECTION_PADDING_X} flex-shrink-0 border-b border-zinc-100 pb-0 lg:pb-4`}
      >
        {props.banner}
        <div className="relative flex w-full items-center justify-between">
          <Breadcrumbs
            items={[
              { name: "My Library", link: "/library" },
              {
                name: props.agent.name,
                link: `/library/agents/${props.agent.id}`,
              },
              ...(props.additionalBreadcrumb
                ? [props.additionalBreadcrumb]
                : []),
            ]}
          />
          <div className="absolute right-0">
            <AgentSettingsModal agent={props.agent} />
          </div>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-visible">
        {props.children}
      </div>
    </SectionWrap>
  );
}
