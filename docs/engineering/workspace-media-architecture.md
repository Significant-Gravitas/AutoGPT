# Workspace & Media File Architecture

This document describes the architecture for handling user files in AutoGPT Platform, covering persistent user storage (Workspace) and ephemeral media processing pipelines.

## Overview

The platform has two distinct file-handling layers:

| Layer | Purpose | Persistence | Scope |
|-------|---------|-------------|-------|
| **Workspace** | Long-term user file storage | Persistent (DB + GCS/local) | Per-user, session-scoped access |
| **Media Pipeline** | Ephemeral file processing for blocks | Temporary (local disk) | Per-execution |

## Database Models

### UserWorkspace

Represents a user's file storage space. Created on-demand (one per user).

```prisma
model UserWorkspace {
  id        String   @id @default(uuid())
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  userId    String   @unique
  Files     UserWorkspaceFile[]
}
```

**Key points:**
- One workspace per user (enforced by `@unique` on `userId`)
- Created lazily via `get_or_create_workspace()` 
- Uses upsert to handle race conditions

### UserWorkspaceFile

Represents a file stored in a user's workspace.

```prisma
model UserWorkspaceFile {
  id          String    @id @default(uuid())
  workspaceId String
  name        String    // User-visible filename
  path        String    // Virtual path (e.g., "/sessions/abc123/image.png")
  storagePath String    // Actual storage path (gcs://... or local://...)
  mimeType    String
  sizeBytes   BigInt
  checksum    String?   // SHA256 for integrity
  isDeleted   Boolean   @default(false)
  deletedAt   DateTime?
  metadata    Json      @default("{}")

  @@unique([workspaceId, path])  // Enforce unique paths within workspace
}
```

**Key points:**
- `path` is a virtual path for organizing files (not actual filesystem path)
- `storagePath` contains the actual GCS or local storage location
- Soft-delete pattern: `isDeleted` flag with `deletedAt` timestamp
- Path is modified on delete to free up the virtual path for reuse

---

## WorkspaceManager

**Location:** `backend/util/workspace.py`

High-level API for workspace file operations. Combines storage backend operations with database record management.

### Initialization

```python
from backend.util.workspace import WorkspaceManager

# Basic usage
manager = WorkspaceManager(user_id="user-123", workspace_id="ws-456")

# With session scoping (CoPilot sessions)
manager = WorkspaceManager(
    user_id="user-123",
    workspace_id="ws-456", 
    session_id="session-789"
)
```

### Session Scoping

When `session_id` is provided, files are isolated to `/sessions/{session_id}/`:

```python
# With session_id="abc123":
manager.write_file(content, "image.png")  
# → stored at /sessions/abc123/image.png

# Cross-session access is explicit:
manager.read_file("/sessions/other-session/file.txt")  # Works
```

**Why session scoping?**
- CoPilot conversations need file isolation
- Prevents file collisions between concurrent sessions
- Allows session cleanup without affecting other sessions

### Core Methods

| Method | Description |
|--------|-------------|
| `write_file(content, filename, path?, mime_type?, overwrite?)` | Write file to workspace |
| `read_file(path)` | Read file by virtual path |
| `read_file_by_id(file_id)` | Read file by ID |
| `list_files(path?, limit?, offset?, include_all_sessions?)` | List files |
| `delete_file(file_id)` | Soft-delete a file |
| `get_download_url(file_id, expires_in?)` | Get signed download URL |
| `get_file_info(file_id)` | Get file metadata |
| `get_file_info_by_path(path)` | Get file metadata by path |
| `get_file_count(path?, include_all_sessions?)` | Count files |

### Storage Backends

WorkspaceManager delegates to `WorkspaceStorageBackend`:

| Backend | When Used | Storage Path Format |
|---------|-----------|---------------------|
| `GCSWorkspaceStorage` | `private_user_data_bucket` (or the legacy bucket fallback) is configured | `gcs://bucket/workspaces/{ws_id}/{file_id}/{filename}` |
| `LocalWorkspaceStorage` | No GCS bucket configured | `local://{ws_id}/{file_id}/{filename}` |

### Public and private media

Hosted storage has two trust boundaries:

- `PRIVATE_USER_DATA_BUCKET` is the default destination for user uploads,
  generated library images, custom Expert avatars, workspaces, transcripts,
  and temporary agent inputs. Private media is read through an authenticated,
  non-cacheable API that serves a file to its owner, admins and members of
  an organization the owner belongs to.
- `PUBLIC_SITE_MEDIA_BUCKET` contains only objects that a trusted publication
  flow explicitly copied after approval, plus media whose purpose is inherently
  public such as OAuth consent-screen logos. Anonymous exact-object reads are
  allowed, but anonymous bucket listing is not.

An upload caller cannot select the public destination. Publication is a
separate privileged operation; sharing a private resource grants access through
its opaque application URL and does not make its storage prefix public.

### Moving a single-bucket deployment to split buckets

Stored rows hold `gcs://<bucket>/...` paths and full GCS URLs, and the storage
code only reads from the configured private bucket, so the existing bucket must
become the private one:

1. Create the new public bucket. Deploy with `PRIVATE_USER_DATA_BUCKET` set to
   the existing `MEDIA_GCS_BUCKET_NAME` bucket and `PUBLIC_SITE_MEDIA_BUCKET`
   set to the new one. Pointing `PRIVATE_USER_DATA_BUCKET` at a new bucket
   instead makes every existing workspace file and transcript unreadable.
   Cloud startup rejects that unsafe partial migration. After every stored
   bucket-qualified path has been migrated to a new private bucket, clear
   `MEDIA_GCS_BUCKET_NAME` before selecting the new private bucket.
   Deploy the frontend first: once the backend has both names set, uploads
   return `/api/store/submissions/media/...`, which a frontend without the new
   rewrite answers with a 404.
   If anonymous users hold `roles/storage.objectViewer` on the old bucket,
   they can list every object in it until step 4. Swap that binding for
   `roles/storage.legacyObjectReader` first: existing links keep working and
   listing stops.
2. Copy everything that is already public to the public bucket and repoint its
   rows: `poetry run python scripts/publish_live_media.py` (dry run), then
   again with `--apply` until it exits 0. A non-zero exit means a live
   reference was not published (copy failure, conflict, an object outside the
   listing's owners or an unrecognised URL) and would break in step 4. This
   covers approved listing media, the avatars of creators with a public
   listing, library copies of listing images and OAuth app logos. It is also
   the repair tool when a copy at approval time failed.
3. Rewrite the remaining private media URLs to the authenticated API path:
   `poetry run python scripts/backfill_private_media_urls.py` (dry run), then
   `--apply`, re-running until it reports no conflicts. It exits 2 while any
   reference stays on the old bucket (held as public, cross-user, ambiguous,
   malformed or unrecognised): those stop loading in step 4, so check the
   counts before going on. Both scripts commit in small batches and can be
   re-run.
4. Run step 2's dry run once more and check it exits 0. Then remove every
   public binding from the old bucket and turn on public access prevention, so
   no object-level grant can expose a file again.

The private media endpoint serves a file to its owner, to admins and to
members of an organization the owner belongs to, so a leaked URL is useless to
anyone else. Published copies are never deleted automatically: when a listing
is taken down or a creator changes their avatar, the old public copy stays in
the public bucket until someone removes it by hand.

Hosted private image uploads are limited to 4 MiB so the frontend proxy can
buffer and deliver the complete authenticated response below Vercel's body
limit. Private videos retain the general 50 MiB upload limit and are delivered
in bounded range responses. An oversized object written before this limit is
served only after the same authorization check, using a 60-second signed URL
that bypasses the frontend proxy without granting bucket listing access.

---

## store_media_file()

**Location:** `backend/util/file.py`

The media normalization pipeline. Handles various input types and normalizes them for processing or output.

### Purpose

Blocks receive files in many formats (URLs, data URIs, workspace references, local paths). `store_media_file()` normalizes these to a consistent format based on what the block needs.

### Input Types Handled

| Input Format | Example | How It's Processed |
|--------------|---------|-------------------|
| Data URI | `data:image/png;base64,iVBOR...` | Decoded, virus scanned, written locally |
| HTTP(S) URL | `https://example.com/image.png` | Downloaded, virus scanned, written locally |
| Workspace URI | `workspace://abc123` or `workspace:///path/to/file` | Read from workspace, virus scanned, written locally |
| Cloud path | `gcs://bucket/path` | Downloaded, virus scanned, written locally |
| Local path | `image.png` | Verified to exist in exec_file directory |

### Return Formats

The `return_format` parameter determines what you get back:

```python
from backend.util.file import store_media_file

# For local processing (ffmpeg, MoviePy, PIL)
local_path = await store_media_file(
    file=input_file,
    execution_context=ctx,
    return_format="for_local_processing"
)
# Returns: "image.png" (relative path in exec_file dir)

# For external APIs (Replicate, OpenAI, etc.)
data_uri = await store_media_file(
    file=input_file,
    execution_context=ctx,
    return_format="for_external_api"
)
# Returns: "data:image/png;base64,iVBOR..."

# For block output (adapts to execution context)
output = await store_media_file(
    file=input_file,
    execution_context=ctx,
    return_format="for_block_output"
)
# In CoPilot: Returns "workspace://file-id#image/png"
# In graphs:  Returns "data:image/png;base64,..."
```

### Execution Context

`store_media_file()` requires an `ExecutionContext` with:
- `graph_exec_id` - Required for temp file location
- `user_id` - Required for workspace access
- `workspace_id` - Optional; enables workspace features
- `session_id` - Optional; for session scoping in CoPilot

---

## Responsibility Boundaries

### Virus Scanning

| Component | Scans? | Notes |
|-----------|--------|-------|
| `store_media_file()` | ✅ Yes | Scans **all** content before writing to local disk |
| `WorkspaceManager.write_file()` | ✅ Yes | Scans content before persisting |

**Scanning happens at:**
1. `store_media_file()` — scans everything it downloads/decodes
2. `WorkspaceManager.write_file()` — scans before persistence

Tools like `WriteWorkspaceFileTool` don't need to scan because `WorkspaceManager.write_file()` handles it.

### Persistence

| Component | Persists To | Lifecycle |
|-----------|-------------|-----------|
| `store_media_file()` | Temp dir (`/tmp/exec_file/{exec_id}/`) | Cleaned after execution |
| `WorkspaceManager` | GCS or local storage + DB | Persistent until deleted |

**Automatic cleanup:** `clean_exec_files(graph_exec_id)` removes temp files after execution completes.

---

## Decision Tree: WorkspaceManager vs store_media_file

```text
┌─────────────────────────────────────────────────────┐
│ What do you need to do with the file?               │
└─────────────────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    Process in a block          Store for user access
    (ffmpeg, PIL, etc.)         (CoPilot files, uploads)
           │                           │
           ▼                           ▼
    store_media_file()           WorkspaceManager
    with appropriate             
    return_format                
           │                           
           │                           
    ┌──────┴──────┐                    
    ▼             ▼                    
 "for_local_   "for_block_
 processing"   output"
    │             │
    ▼             ▼
 Get local    Auto-saves to
 path for     workspace in
 tools        CoPilot context

Store for user access
    │
    ├── write_file() ─── Upload + persist (scans internally)
    ├── read_file() / get_download_url() ─── Retrieve
    └── list_files() / delete_file() ─── Manage
```

### Quick Reference

| Scenario | Use |
|----------|-----|
| Block needs to process a file with ffmpeg | `store_media_file(..., return_format="for_local_processing")` |
| Block needs to send file to external API | `store_media_file(..., return_format="for_external_api")` |
| Block returning a generated file | `store_media_file(..., return_format="for_block_output")` |
| API endpoint handling file upload | `WorkspaceManager.write_file()` (handles virus scanning internally) |
| API endpoint serving file download | `WorkspaceManager.get_download_url()` |
| Listing user's files | `WorkspaceManager.list_files()` |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `backend/data/workspace.py` | Database CRUD operations for UserWorkspace and UserWorkspaceFile |
| `backend/util/workspace.py` | `WorkspaceManager` class - high-level workspace API |
| `backend/util/workspace_storage.py` | Storage backends (GCS, local) and `WorkspaceStorageBackend` interface |
| `backend/util/file.py` | `store_media_file()` and media processing utilities |
| `backend/util/virus_scanner.py` | `VirusScannerService` and `scan_content_safe()` |
| `schema.prisma` | Database model definitions |

---

## Common Patterns

### Block Processing a User's File

```python
async def run(self, input_data, *, execution_context, **kwargs):
    # Normalize input to local path
    local_path = await store_media_file(
        file=input_data.video,
        execution_context=execution_context,
        return_format="for_local_processing",
    )
    
    # Process with local tools
    output_path = process_video(local_path)
    
    # Return (auto-saves to workspace in CoPilot)
    result = await store_media_file(
        file=output_path,
        execution_context=execution_context,
        return_format="for_block_output",
    )
    yield "output", result
```

### API Upload Endpoint

```python
from backend.util.virus_scanner import VirusDetectedError, VirusScanError

async def upload_file(file: UploadFile, user_id: str, workspace_id: str):
    content = await file.read()

    # write_file handles virus scanning internally
    manager = WorkspaceManager(user_id, workspace_id)
    try:
        workspace_file = await manager.write_file(
            content=content,
            filename=file.filename,
        )
    except VirusDetectedError:
        raise HTTPException(status_code=400, detail="File rejected: virus detected")
    except VirusScanError:
        raise HTTPException(status_code=503, detail="Virus scanning unavailable")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"file_id": workspace_file.id}
```

---

## Configuration

| Setting | Purpose | Default |
|---------|---------|---------|
| `private_user_data_bucket` | Private GCS bucket for user media, workspace storage, transcripts, and temporary uploads | Falls back to `media_gcs_bucket_name`, then local storage |
| `public_site_media_bucket` | Public GCS bucket for explicitly published marketplace media and OAuth app logos | Falls back to `media_gcs_bucket_name` for compatibility |
| `media_gcs_bucket_name` | Legacy single-bucket setting for self-hosted compatibility | None |
| `workspace_storage_dir` | Local storage directory | `{app_data}/workspaces` |
| `max_file_size_mb` | Maximum file size in MB | 100 |
| `clamav_service_enabled` | Enable virus scanning | true |
| `clamav_service_host` | ClamAV daemon host | localhost |
| `clamav_service_port` | ClamAV daemon port | 3310 |
| `clamav_max_concurrency` | Max concurrent scans to ClamAV daemon | 5 |
| `clamav_mark_failed_scans_as_clean` | If true, scan failures pass content through instead of rejecting (⚠️ security risk if ClamAV is unreachable) | false |
