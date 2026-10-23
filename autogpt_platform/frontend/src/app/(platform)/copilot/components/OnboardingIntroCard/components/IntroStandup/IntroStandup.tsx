"use client";

import { ExpertConnectServiceDialog } from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/ExpertConnectServiceDialog";
import { AskCard } from "./components/AskCard";
import { TeamCard } from "./components/TeamCard";
import { joinNames } from "./helpers";
import { useIntroStandup } from "./useIntroStandup";

interface Props {
  path: string;
  disabled: boolean;
  onAnswer: (message: string) => unknown;
}

/** The rest of Otto's first turn after the greeting: the team he hired for
 *  the user, each hire's day-one job and setup status, and one ask. */
export function IntroStandup({ path, disabled, onAnswer }: Props) {
  const {
    team,
    ask,
    isConnectOpen,
    openConnect,
    closeConnect,
    handleConnected,
    dismissConnect,
    startVoice,
    focusComposer,
    answer,
    canRecord,
  } = useIntroStandup({ path, onAnswer });

  return (
    <>
      {team.length > 0 && <TeamCard team={team} />}
      {ask && (
        <AskCard
          ask={ask}
          disabled={disabled}
          onConnect={openConnect}
          onDismissConnect={dismissConnect}
          onAnswer={answer}
          onStartVoice={startVoice}
          onType={focusComposer}
          canRecord={canRecord}
        />
      )}
      {ask?.kind === "connect" && isConnectOpen && (
        <ExpertConnectServiceDialog
          open
          expertName={joinNames(ask.names)}
          initialProviderId={ask.provider}
          onClose={closeConnect}
          onConnected={handleConnected}
        />
      )}
    </>
  );
}
