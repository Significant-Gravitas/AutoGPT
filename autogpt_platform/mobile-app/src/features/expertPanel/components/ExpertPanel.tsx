import { useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { FileLibrary } from "@/features/files/components/FileLibrary";
import { createStyles, spacing } from "@/lib/theme";

import { ActivityTab } from "./ActivityTab";
import { PanelTopBar } from "./PanelTopBar";
import { PanelTabs, type PanelTab } from "./PanelTabs";

interface Props {
  sessionID: string;
  expertID: string;
  name: string;
  role: string | null;
  avatar: string | number | null;
  tint: string | null;
  isOpen: boolean;
  onClose: () => void;
}

/** The panel in an expert's chat: who they are, what they do, their files. */
export function ExpertPanel({
  sessionID,
  expertID,
  name,
  role,
  avatar,
  tint,
  isOpen,
  onClose,
}: Props) {
  const [tab, setTab] = useState<PanelTab>("activity");
  const insets = useSafeAreaInsets();
  const styles = useStyles();

  return (
    <ScrollView
      contentContainerStyle={[
        styles.content,
        { paddingTop: insets.top + spacing.xs, paddingBottom: insets.bottom + spacing.xl },
      ]}
    >
      <PanelTopBar sessionID={sessionID} onClose={onClose} />
      <View style={styles.identity}>
        <ExpertAvatar name={name} source={avatar} size={76} tint={tint} />
        <Text accessibilityRole="header" numberOfLines={1} style={styles.name}>
          {name}
        </Text>
        {role ? (
          <Text numberOfLines={1} style={styles.role}>
            {role}
          </Text>
        ) : null}
      </View>
      <PanelTabs selected={tab} onSelect={setTab} />
      {tab === "activity" ? (
        <ActivityTab expertID={expertID} expertName={name} isOpen={isOpen} />
      ) : (
        <FileLibrary
          sessionID={sessionID}
          expert={{ id: expertID, name }}
          isOpen={isOpen}
        />
      )}
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  content: { gap: spacing.xl, paddingHorizontal: spacing.sm },
  identity: { alignItems: "center", gap: spacing.xs, marginTop: -spacing.sm },
  name: {
    fontSize: 20,
    lineHeight: 26,
    fontWeight: "600",
    color: colors.text,
    marginTop: spacing.sm,
  },
  role: { fontSize: 13, lineHeight: 18, color: colors.textTertiary },
}));
