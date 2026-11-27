import { LinkSquare02Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, ScrollView, Text, TextInput, View, useWindowDimensions } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { logoID } from "../../helpers";
import { actionLabel, signInServerURL } from "../../listHelpers";
import { BottomSheet } from "../BottomSheet";
import { ConnectionRow } from "../ConnectionRow";
import { CredentialForm } from "../CredentialSheet/CredentialSheet";
import { ServerStep } from "./components/ServerStep";
import { SheetSection } from "./components/SheetSection";
import { SERVER_ROW } from "./helpers";
import { useAddConnectionSheet } from "./useAddConnectionSheet";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  sessionID: string;
  /** Set in an expert's chat: their accounts can be allowed, and new ones are. */
  expertID: string | null;
}

/** Add a connection by hand: allow an account you have, or connect a new one. */
export function AddConnectionSheet({ isOpen, onClose, sessionID, expertID }: Props) {
  return (
    <BottomSheet isOpen={isOpen} onClose={onClose}>
      {/* Mounted only while open, so each opening starts fresh. */}
      {isOpen ? <SheetBody sessionID={sessionID} expertID={expertID} onDone={onClose} /> : null}
    </BottomSheet>
  );
}

interface BodyProps {
  sessionID: string;
  expertID: string | null;
  onDone: () => void;
}

function SheetBody({ sessionID, expertID, onDone }: BodyProps) {
  const sheet = useAddConnectionSheet({ sessionID, expertID, onDone });
  const { connect, lists } = sheet;
  const styles = useStyles();
  const colors = useColors();
  const { height } = useWindowDimensions();

  if (connect.sheet) {
    return (
      <View style={styles.body}>
        <CredentialForm key={`${connect.sheet.rowKey}:${connect.sheet.kind}`} target={connect.sheet} onSaved={connect.handleSaved} />
        <Button label="Back" onPress={connect.closeSheet} variant="ghost" size="small" />
      </View>
    );
  }

  if (sheet.isAskingServer) {
    return (
      <ServerStep
        value={sheet.serverText}
        onChange={sheet.setServerText}
        canSubmit={sheet.canSignInToServer && !sheet.isBusy}
        isBusy={sheet.isRowBusy(SERVER_ROW)}
        error={connect.errors[SERVER_ROW] ?? null}
        onSubmit={sheet.signInToServer}
        onBack={sheet.cancelServer}
      />
    );
  }

  function renderServices() {
    if (sheet.hasServicesFailed) {
      return (
        <View style={styles.notice}>
          <Text style={styles.noticeText}>{"Couldn't load the services."}</Text>
          <Button label="Try again" onPress={sheet.retryProviders} variant="secondary" size="small" />
        </View>
      );
    }
    if (sheet.isLoadingServices) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    if (lists.services.length === 0 && !lists.showServer) return null;
    return (
      <SheetSection title="New">
        {lists.services.map((service) => (
          <ConnectionRow
            key={service.key}
            name={service.name}
            detail={service.action.kind === "web" ? "Finish this on the web" : service.description}
            logoID={logoID(service.key)}
            error={connect.errors[service.key]}
            action={{
              label: connect.usesToken(service.key, signInServerURL(service.action))
                ? "Add token"
                : actionLabel(service.action),
              variant: service.action.kind === "web" ? "ghost" : "outline",
              onPress: () => sheet.connectService(service),
              isBusy: sheet.isRowBusy(service.key),
              isDisabled: sheet.isBusy,
            }}
          />
        ))}
        {lists.showServer ? (
          <ConnectionRow
            name="MCP server"
            detail="Any server by URL"
            action={{ label: "Add", variant: "outline", onPress: sheet.askServer, isDisabled: sheet.isBusy }}
          />
        ) : null}
      </SheetSection>
    );
  }

  return (
    <View style={styles.body}>
      <Text accessibilityRole="header" style={styles.title}>
        Add connection
      </Text>
      <View style={styles.search}>
        <Icon icon={Search01Icon} size={18} color={colors.textTertiary} />
        <TextInput
          value={sheet.query}
          onChangeText={sheet.setQuery}
          placeholder="Search services"
          placeholderTextColor={colors.textTertiary}
          accessibilityLabel="Search services"
          autoCorrect={false}
          autoCapitalize="none"
          returnKeyType="search"
          style={styles.searchInput}
        />
      </View>
      <ScrollView
        style={{ maxHeight: height * 0.55 }}
        contentContainerStyle={styles.lists}
        keyboardShouldPersistTaps="handled"
      >
        {lists.accounts.length > 0 ? (
          <SheetSection title="Already connected">
            {lists.accounts.map((account) => {
              const rowKey = `account:${account.id}`;
              return (
                <ConnectionRow
                  key={account.id}
                  name={account.name}
                  detail={account.detail}
                  logoID={account.logoID}
                  error={connect.errors[rowKey]}
                  action={{
                    label: "Allow",
                    variant: "outline",
                    onPress: () => sheet.allowAccount(account.id),
                    isBusy: sheet.isRowBusy(rowKey),
                    isDisabled: sheet.isBusy,
                  }}
                />
              );
            })}
          </SheetSection>
        ) : null}
        {renderServices()}
        {lists.isEmpty && !sheet.isLoadingServices && !sheet.hasServicesFailed ? (
          <Text style={styles.noticeText}>{`Nothing matches "${sheet.query.trim()}".`}</Text>
        ) : null}
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
      </ScrollView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  title: { ...type.heading, color: colors.text },
  search: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  searchInput: { ...type.body, flex: 1, color: colors.text, paddingVertical: spacing.sm + 2 },
  lists: { gap: spacing.lg, paddingBottom: spacing.sm },
  loading: { paddingVertical: spacing.xl },
  notice: { alignItems: "flex-start", gap: spacing.sm },
  noticeText: { ...type.small, color: colors.textSecondary },
}));
