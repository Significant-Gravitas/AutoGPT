import { shortID } from "../helpers";
import type { SyncMapping } from "../types";

interface Props {
  mapping: SyncMapping | null;
  syncing: boolean;
  onSync: () => Promise<void>;
}

export function SyncPanel({ mapping, syncing, onSync }: Props) {
  return (
    <section className="sync-panel">
      <div className="panel-heading">
        <div>
          <p>Identity bridge</p>
          <h2>AutoGPT tenant mapping</h2>
        </div>
        <span className={mapping ? "sync-status synced" : "sync-status"}>
          {mapping ? "Synced" : "Not synced"}
        </span>
      </div>
      {mapping ? (
        <dl>
          <div>
            <dt>User</dt>
            <dd title={mapping.autoGPTUserID}>
              {shortID(mapping.autoGPTUserID)}
            </dd>
          </div>
          <div>
            <dt>Organization</dt>
            <dd title={mapping.autoGPTOrganizationID}>
              {shortID(mapping.autoGPTOrganizationID)}
            </dd>
          </div>
          <div>
            <dt>Team</dt>
            <dd title={mapping.autoGPTTeamID}>
              {shortID(mapping.autoGPTTeamID)}
            </dd>
          </div>
        </dl>
      ) : (
        <p className="empty-sync">
          The first sync or chat request will JIT-provision this tenant.
        </p>
      )}
      <button disabled={syncing} onClick={() => void onSync()}>
        {syncing ? "Syncing…" : mapping ? "Refresh mapping" : "Sync to AutoGPT"}
      </button>
    </section>
  );
}
