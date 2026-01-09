import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { AgentSettingsButton } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/other/AgentSettingsButton";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../helpers";
import { SectionWrap } from "../other/SectionWrap";

interface Props {
  agent: LibraryAgent;
  children: React.ReactNode;
  banner?: React.ReactNode;
  additionalBreadcrumb?: { name: string; link?: string };
  onSelectSettings?: () => void;
  selectedSettings?: boolean;
}

export function SelectedViewLayout(props: Props) {
  return (
    <SectionWrap className="relative mb-3 flex min-h-0 flex-1 flex-col">
      <div
        className={`${AGENT_LIBRARY_SECTION_PADDING_X} flex-shrink-0 border-b border-zinc-100 pb-0 lg:pb-4`}
      >
        {props.banner && <div className="mb-4">{props.banner}</div>}
        <div className="relative flex w-fit items-center gap-2">
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
          {props.agent && props.onSelectSettings && (
            <div className="absolute -right-8">
              <AgentSettingsButton
                agent={props.agent}
                onSelectSettings={props.onSelectSettings}
                selected={props.selectedSettings}
              />
            </div>
          )}
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-visible">
        {props.children}
      </div>
    </SectionWrap>
  );
}
