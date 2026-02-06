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
| `get_file_count(path?, include_all_sessions?)` | Count files |

### Storage Backends

WorkspaceManager delegates to `WorkspaceStorageBackend`:

| Backend | When Used | Storage Path Format |
|---------|-----------|---------------------|
| `GCSWorkspaceStorage` | `media_gcs_bucket_name` is configured | `gcs://bucket/workspaces/{ws_id}/{file_id}/{filename}` |
| `LocalWorkspaceStorage` | No GCS bucket configured | `local://{ws_id}/{file_id}/{filename}` |

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

```
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
```

### Quick Reference

| Scenario | Use |
|----------|-----|
| Block needs to process a file with ffmpeg | `store_media_file(..., return_format="for_local_processing")` |
| Block needs to send file to external API | `store_media_file(..., return_format="for_external_api")` |
| Block returning a generated file | `store_media_file(..., return_format="for_block_output")` |
| API endpoint handling file upload | `WorkspaceManager.write_file()` (after virus scan) |
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
async def upload_file(file: UploadFile, user_id: str, workspace_id: str):
    content = await file.read()
    
    # write_file handles virus scanning
    manager = WorkspaceManager(user_id, workspace_id)
    workspace_file = await manager.write_file(
        content=content,
        filename=file.filename,
    )
    
    return {"file_id": workspace_file.id}
```

---

## Configuration

| Setting | Purpose | Default |
|---------|---------|---------|
| `media_gcs_bucket_name` | GCS bucket for workspace storage | None (uses local) |
| `workspace_storage_dir` | Local storage directory | `{app_data}/workspaces` |
| `max_file_size_mb` | Maximum file size in MB | 100 |
| `clamav_service_enabled` | Enable virus scanning | true |
| `clamav_service_host` | ClamAV daemon host | localhost |
| `clamav_service_port` | ClamAV daemon port | 3310 |
