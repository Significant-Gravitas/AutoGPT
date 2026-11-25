// The backend's rule for a draft save (`_save_wizard_progress` in
// backend/data/onboarding.py), as these tests in
// backend/data/onboarding_wizard_db_test.py pin it against Postgres:
// test_a_tabs_next_save_is_kept_over_its_own_unanswered_one,
// test_the_same_save_arriving_twice_is_not_refused,
// test_an_older_save_landing_late_cannot_replace_a_newer_one,
// test_another_tab_saving_from_the_same_revision_is_refused and
// test_a_tabs_stale_save_is_refused_once_anything_else_saved.

export interface StoredDraft<Progress> {
  progress: Progress | null;
  revision: number;
  writerId: string | null;
  writeSeq: number | null;
}

interface DraftSave<Progress> {
  wizardProgress?: Progress | null;
  wizardRevision?: number | null;
  wizardWriterId?: string | null;
  wizardWriteSeq?: number | null;
}

// The draft the server holds after the save, or null when it refuses it.
export function landDraftSave<Progress>(
  stored: StoredDraft<Progress>,
  save: DraftSave<Progress>,
): StoredDraft<Progress> | null {
  const fromSameTab =
    save.wizardWriterId != null &&
    save.wizardWriterId === stored.writerId &&
    (stored.writeSeq ?? Infinity) <= (save.wizardWriteSeq ?? 0);
  if (save.wizardRevision !== stored.revision && !fromSameTab) return null;
  return {
    progress: save.wizardProgress ?? null,
    revision: stored.revision + 1,
    writerId: save.wizardWriterId ?? null,
    writeSeq: save.wizardWriteSeq ?? null,
  };
}
