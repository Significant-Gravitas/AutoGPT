import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { ActivitySection } from "@/features/expertPanel/components/ActivitySection";
import { RetryNotice } from "@/features/files/components/RetryNotice";
import { createStyles, spacing, type } from "@/lib/theme";

import { AddConnectionSheet } from "../AddConnectionSheet/AddConnectionSheet";
import { ConnectionRow } from "../ConnectionRow";
import { useYourConnections } from "./useYourConnections";

interface Props {
  sessionID: string;
  isOpen: boolean;
}

/**
 * Every account you have connected, under the files of a chat with Otto.
 * Removing one is a Settings job, so the rows have no button.
 */
export function YourConnections({ sessionID, isOpen }: Props) {
  const section = useYourConnections(isOpen);
  const styles = useStyles();

  function renderList() {
    if (section.hasFailed) {
      return <RetryNotice message="Couldn't load your connections." onRetry={section.retry} />;
    }
    if (!section.isLoading && section.accounts.length === 0) {
      return <Text style={styles.empty}>Nothing connected yet.</Text>;
    }
    return (
      <ActivitySection title="Your connections" isLoading={section.isLoading}>
        {section.accounts.map((account) => (
          <ConnectionRow key={account.id} name={account.name} detail={account.detail} logoID={account.logoID} />
        ))}
      </ActivitySection>
    );
  }

  return (
    <View style={styles.section}>
      <Text accessibilityRole="header" style={styles.title}>
        Connections
      </Text>
      {renderList()}
      <Button
        label="Add connection"
        onPress={section.openAdd}
        variant="outline"
        leadingIcon={PlusSignIcon}
        fullWidth
      />
      <AddConnectionSheet isOpen={section.isAdding} onClose={section.closeAdd} sessionID={sessionID} expertID={null} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.md },
  title: { ...type.heading, color: colors.text, paddingHorizontal: spacing.md },
  empty: { ...type.small, color: colors.textTertiary, paddingHorizontal: spacing.md },
}));
