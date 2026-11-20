import { LinkSquare02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { PanelEmptyState } from "@/components/PanelEmptyState";
import { ActivitySection } from "@/features/expertPanel/components/ActivitySection";
import { RetryNotice } from "@/features/files/components/RetryNotice";
import { createStyles, spacing, type } from "@/lib/theme";

import { AddConnectionSheet } from "../AddConnectionSheet/AddConnectionSheet";
import { ConnectionRow } from "../ConnectionRow";
import { CredentialSheet } from "../CredentialSheet/CredentialSheet";
import { ProviderTile } from "../ProviderTile";
import { useExpertConnections } from "./useExpertConnections";

interface Props {
  sessionID: string;
  expertID: string;
  expertName: string;
  isOpen: boolean;
}

/** The Connections tab: what the expert can use, and what their workflows still need. */
export function ExpertConnections({ sessionID, expertID, expertName, isOpen }: Props) {
  const tab = useExpertConnections({ sessionID, expertID, expertName, isOpen });
  const { connect } = tab;
  const styles = useStyles();

  function renderBody() {
    if (tab.hasFailed) {
      return <RetryNotice message="Couldn't load these connections." onRetry={tab.retry} />;
    }
    if (tab.isLoading) {
      return <ActivitySection title={`${expertName} can use`} isLoading>{null}</ActivitySection>;
    }
    if (tab.isEmpty) {
      return (
        <PanelEmptyState
          tiles={[
            <ProviderTile key="slack" name="Slack" logoID="slack" size={36} />,
            <ProviderTile key="notion" name="Notion" logoID="notion" size={36} />,
            <ProviderTile key="google" name="Google" logoID="google" size={36} />,
          ]}
          title="No connections yet"
          description={`Connect a service and ${expertName} can use it for you.`}
        >
          <Button label="Add connection" onPress={tab.openAdd} size="small" leadingIcon={PlusSignIcon} />
        </PanelEmptyState>
      );
    }
    return (
      <>
        {tab.accounts.length > 0 ? (
          <ActivitySection title={`${expertName} can use`} isLoading={false}>
            {tab.accounts.map((account) => (
              <ConnectionRow
                key={account.id}
                name={account.name}
                detail={account.detail}
                logoID={account.logoID}
                error={tab.removeError(account)}
                action={{
                  label: "Remove",
                  variant: "ghost",
                  onPress: () => tab.confirmRemove(account),
                  isBusy: tab.isRemoving(account),
                  isDisabled: tab.isBusy,
                }}
              />
            ))}
          </ActivitySection>
        ) : null}
        {tab.needs.length > 0 ? (
          <ActivitySection title={`${expertName}'s workflows also need`} isLoading={false}>
            {tab.needs.map((need) => (
              <ConnectionRow
                key={need.key}
                name={need.name}
                detail={need.detail}
                logoID={need.logoID}
                error={connect.errors[need.key]}
                action={
                  need.label
                    ? {
                        label: need.usesToken ? "Add token" : need.label,
                        variant: need.label === "Sign in" && !need.usesToken ? "primary" : "outline",
                        onPress: () => tab.handleNeed(need),
                        isBusy: tab.isNeedBusy(need),
                        isDisabled: tab.isBusy,
                      }
                    : null
                }
              />
            ))}
          </ActivitySection>
        ) : null}
        <Button
          label="Add connection"
          onPress={tab.openAdd}
          variant="outline"
          leadingIcon={PlusSignIcon}
          fullWidth
          disabled={tab.isBusy}
        />
      </>
    );
  }

  return (
    <View style={styles.tab}>
      {renderBody()}
      {connect.needsPlan ? (
        <View style={styles.notice}>
          <Text style={styles.noticeText}>Connecting a service needs an active plan.</Text>
          <Button
            label="Finish this on the web"
            onPress={connect.openOnWeb}
            variant="outline"
            size="small"
            leadingIcon={LinkSquare02Icon}
          />
        </View>
      ) : null}
      <CredentialSheet target={connect.sheet} onClose={connect.closeSheet} onSaved={connect.handleSaved} />
      <AddConnectionSheet isOpen={tab.isAdding} onClose={tab.closeAdd} sessionID={sessionID} expertID={expertID} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  tab: { gap: spacing.xl },
  notice: { alignItems: "flex-start", gap: spacing.sm, paddingHorizontal: spacing.md },
  noticeText: { ...type.small, color: colors.textSecondary },
}));
