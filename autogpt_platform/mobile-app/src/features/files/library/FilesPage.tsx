import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, FlatList, RefreshControl, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { FLOATING_HEADER_HEIGHT, FloatingHeader } from "@/components/FloatingHeader";
import { IconButton } from "@/components/IconButton";
import { MarketplaceSearchField } from "@/features/marketplace/components/MarketplaceSearchField";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { FileRow } from "../components/FileRow";
import { FileViewer } from "../components/FileViewer";
import { RetryNotice } from "../components/RetryNotice";
import { FolderList } from "./components/FolderList";
import { OwnerChips } from "./components/OwnerChips";
import { PageTop } from "./components/PageTop";
import { SectionHeading } from "./components/SectionHeading";
import { useFilesPage } from "./useFilesPage";

interface Props {
  /** Null for the top level. */
  folderID: string | null;
}

/** The web's Files page: search, who made it, folders, then files that load as you scroll. */
export function FilesPage({ folderID }: Props) {
  const page = useFilesPage(folderID);
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();
  const headerHeight = insets.top + FLOATING_HEADER_HEIGHT;
  const folderName = page.path[page.path.length - 1]?.name;

  if (page.isMissingFolder) {
    return (
      <View style={styles.screen}>
        <View style={[styles.missing, { paddingTop: headerHeight + spacing.xl }]}>
          <Text style={styles.missingText}>This folder isn&apos;t here any more.</Text>
          <Button label="Back to Files" variant="secondary" onPress={() => page.openFolder(null)} />
        </View>
        <FloatingHeader fadeColor={colors.panelBackground} />
      </View>
    );
  }

  function renderEmpty() {
    if (page.isLoading) {
      return (
        <View style={[styles.card, styles.placeholder]}>
          <View style={styles.placeholderLine} />
        </View>
      );
    }
    if (page.hasFailed) return <RetryNotice message="Couldn't load your files." onRetry={page.refresh} />;
    if (page.folders.length > 0 || page.haveFoldersFailed) return undefined;
    return (
      <Text style={styles.empty}>
        {page.search.trim() ? `No files match “${page.search.trim()}”.` : "No files here yet."}
      </Text>
    );
  }

  function renderFile({ item, index }: { item: WorkspaceFile; index: number }) {
    const isLast = index === page.files.length - 1;
    return (
      <View style={[styles.segment, index === 0 && styles.first, isLast && styles.last]}>
        <FileRow file={item} isFirst={index === 0} owner={page.ownerOf(item)} onPress={page.setOpenFile} />
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <FlatList
        data={page.files}
        keyExtractor={(file) => file.id}
        renderItem={renderFile}
        onEndReached={page.loadMore}
        onEndReachedThreshold={0.6}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="on-drag"
        contentContainerStyle={{ paddingTop: headerHeight, paddingBottom: insets.bottom + spacing.xxl }}
        refreshControl={
          <RefreshControl
            refreshing={page.isRefreshing}
            onRefresh={page.refresh}
            tintColor={colors.textTertiary}
            progressViewOffset={headerHeight}
          />
        }
        ListHeaderComponent={
          <View>
            <PageTop
              path={page.path}
              allFolders={page.allFolders}
              storage={page.storage}
              onOpenFolder={page.openFolder}
            />
            <MarketplaceSearchField
              value={page.search}
              onChangeText={page.changeSearch}
              placeholder={folderName ? `Search in ${folderName}` : "Search files"}
            />
            <OwnerChips chip={page.chip} team={page.team} onChange={page.setChip} />
            {page.haveFoldersFailed ? (
              <View style={styles.folderNotice}>
                <RetryNotice message="Couldn't load your folders." onRetry={page.refresh} />
              </View>
            ) : null}
            <FolderList folders={page.folders} allFolders={page.allFolders} onOpen={page.openFolder} />
            {page.files.length > 0 ? (
              <SectionHeading title="Files" count={page.hasMore ? null : page.files.length} />
            ) : null}
          </View>
        }
        ListEmptyComponent={renderEmpty()}
        ListFooterComponent={
          page.isLoadingMore ? <ActivityIndicator style={styles.more} color={colors.textTertiary} /> : undefined
        }
      />
      <FloatingHeader
        left={
          folderID ? (
            <IconButton icon={ArrowLeft01Icon} label="Back" onPress={page.goUp} variant="floating" />
          ) : undefined
        }
        fadeColor={colors.panelBackground}
      />
      <FileViewer file={page.openFile} onClose={() => page.setOpenFile(null)} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.panelBackground },
  segment: { marginHorizontal: 14, backgroundColor: colors.panelCard, overflow: "hidden" },
  first: { borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg },
  last: { borderBottomLeftRadius: radius.lg, borderBottomRightRadius: radius.lg },
  card: { marginHorizontal: 14, borderRadius: radius.lg, backgroundColor: colors.panelCard },
  placeholder: { paddingHorizontal: spacing.md, paddingVertical: spacing.lg },
  placeholderLine: { height: 14, width: "60%", borderRadius: 7, backgroundColor: colors.surfacePressed },
  empty: { fontSize: 14, lineHeight: 20, color: colors.textTertiary, paddingHorizontal: 18 },
  more: { marginTop: spacing.lg },
  folderNotice: { marginBottom: spacing.lg },
  missing: { alignItems: "center", gap: spacing.lg, padding: spacing.xl },
  missingText: { fontSize: 16, lineHeight: 23, color: colors.textSecondary, textAlign: "center" },
}));
