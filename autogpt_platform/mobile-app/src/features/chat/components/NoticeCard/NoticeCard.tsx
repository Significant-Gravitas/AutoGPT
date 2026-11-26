import {
  Alert01Icon,
  CheckmarkCircle02Icon,
  Clock01Icon,
  CreditCardIcon,
  PauseIcon,
  PlugSocketIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import type { ToastState } from "@/components/useToast";
import { AddConnectionSheet } from "@/features/connections/components/AddConnectionSheet/AddConnectionSheet";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import type { ChatNotice, NoticeAction } from "../../notice";
import { RunLessOftenSheet } from "./components/RunLessOftenSheet";
import { useNoticeCard } from "./useNoticeCard";

interface Props {
  notice: ChatNotice;
  sessionID: string;
  expertID: string | null;
  onToast: (toast: ToastState) => void;
}

const REASON_ICONS: Record<string, IconSvgElement> = {
  usage: Clock01Icon,
  no_plan: CreditCardIcon,
  codex_access: PlugSocketIcon,
  connection: PlugSocketIcon,
  chat_link: PlugSocketIcon,
  budget: PauseIcon,
  error: Alert01Icon,
  resumed: CheckmarkCircle02Icon,
};

/** Why a scheduled chat was skipped or stopped, with the fix on a button. */
export function NoticeCard({ notice, sessionID, expertID, onToast }: Props) {
  const card = useNoticeCard({ notice, sessionID, onToast });
  const styles = useStyles();
  const colors = useColors();
  const actions = notice.actions.filter((action) => action.kind !== "resume" || !card.isResumed);

  function renderAction(action: NoticeAction, index: number) {
    const common = { label: action.label, variant: "secondary", size: "small" } as const;
    switch (action.kind) {
      case "link":
        return (
          <Button key={`link-${index}-${action.href}`} {...common} onPress={() => card.openLink(action.href)} />
        );
      case "reconnect":
        return (
          <Button
            key={`reconnect-${index}-${action.provider}`}
            {...common}
            onPress={() => card.openConnect(action.provider)}
          />
        );
      case "run_less_often":
        return <Button key={`run-less-often-${index}`} {...common} onPress={card.openLessOften} />;
      case "resume":
        return (
          <Button key={`resume-${index}`} {...common} onPress={card.handleResume} isLoading={card.isResuming} />
        );
    }
  }

  return (
    <View style={styles.card}>
      <View style={styles.row}>
        <Icon
          icon={REASON_ICONS[notice.reason] ?? Clock01Icon}
          size={18}
          color={colors.textSecondary}
        />
        <View style={styles.text}>
          <Text style={styles.headline}>{notice.headline}</Text>
          <Text style={styles.body}>{notice.body}</Text>
        </View>
      </View>
      {actions.length > 0 ? <View style={styles.actions}>{actions.map(renderAction)}</View> : null}
      {notice.scheduleKey ? (
        <RunLessOftenSheet
          scheduleKey={notice.scheduleKey}
          isOpen={card.isLessOftenOpen}
          onClose={card.closeLessOften}
          onSaved={card.handleCadenceSaved}
        />
      ) : null}
      <AddConnectionSheet
        isOpen={card.connectProvider !== null}
        onClose={card.closeConnect}
        sessionID={sessionID}
        expertID={expertID}
        initialProvider={card.connectProvider ?? undefined}
        onConnected={card.handleConnected}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    gap: spacing.md,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.surface,
  },
  row: { flexDirection: "row", alignItems: "flex-start", gap: spacing.sm },
  text: { flex: 1, gap: 2 },
  headline: { ...type.label, color: colors.text },
  body: { ...type.small, color: colors.textSecondary },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm, paddingLeft: 18 + spacing.sm },
}));
