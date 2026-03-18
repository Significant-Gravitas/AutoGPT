"use client";

import type { BulkInvitedUsersResponse } from "@/app/api/__generated__/models/bulkInvitedUsersResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { type ChangeEvent, type FormEvent, useState } from "react";

const MAX_BULK_INVITE_FILE_BYTES = 1024 * 1024;

function getFileSizeError(file: File) {
  if (file.size > MAX_BULK_INVITE_FILE_BYTES) {
    return "Invite file exceeds the maximum size of 1 MB";
  }

  return null;
}

interface Props {
  selectedFile: File | null;
  inputKey: number;
  isSubmitting: boolean;
  lastResult: BulkInvitedUsersResponse | null;
  onFileChange: (file: File | null) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

function getStatusVariant(status: "CREATED" | "SKIPPED" | "ERROR") {
  if (status === "CREATED") {
    return "success";
  }

  if (status === "ERROR") {
    return "error";
  }

  return "info";
}

export function BulkInviteForm({
  selectedFile,
  inputKey,
  isSubmitting,
  lastResult,
  onFileChange,
  onSubmit,
}: Props) {
  const [fileError, setFileError] = useState<string | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.item(0) ?? null;
    if (!file) {
      setFileError(null);
      onFileChange(null);
      return;
    }

    const nextFileError = getFileSizeError(file);
    if (nextFileError) {
      setFileError(nextFileError);
      onFileChange(null);
      event.target.value = "";
      return;
    }

    setFileError(null);
    onFileChange(file);
  }

  return (
    <form className="flex flex-col gap-4" onSubmit={onSubmit}>
      <div className="flex flex-col gap-1">
        <h2 className="text-xl font-semibold text-zinc-900">Bulk invite</h2>
        <p className="text-sm text-zinc-600">
          Upload a <span className="font-medium text-zinc-800">.txt</span> file
          with one email per line, or a{" "}
          <span className="font-medium text-zinc-800">.csv</span> with
          <span className="font-medium text-zinc-800"> email</span> and optional
          <span className="font-medium text-zinc-800"> name</span> columns.
        </p>
      </div>

      <label
        htmlFor="bulk-invite-file-input"
        className="flex cursor-pointer flex-col gap-2 rounded-2xl border border-dashed border-zinc-300 bg-zinc-50 px-4 py-5 text-sm text-zinc-600 transition-colors focus-within:ring-2 focus-within:ring-zinc-500 focus-within:ring-offset-2 hover:border-zinc-400 hover:bg-zinc-100"
      >
        <span className="font-medium text-zinc-900">
          {selectedFile ? selectedFile.name : "Choose invite file"}
        </span>
        <span>Maximum 5,000 users and 1 MB per upload.</span>
        <input
          id="bulk-invite-file-input"
          key={inputKey}
          type="file"
          accept=".txt,.csv,text/plain,text/csv"
          disabled={isSubmitting}
          className="sr-only"
          onChange={handleFileChange}
        />
      </label>

      {fileError ? <p className="text-sm text-red-600">{fileError}</p> : null}

      <Button
        type="submit"
        variant="primary"
        loading={isSubmitting}
        disabled={!selectedFile}
        className="w-full"
      >
        {isSubmitting ? "Uploading invites..." : "Upload invite file"}
      </Button>

      {lastResult ? (
        <div className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-zinc-50 p-4">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded-xl bg-white px-3 py-2">
              <div className="text-lg font-semibold text-zinc-900">
                {lastResult.created_count}
              </div>
              <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">
                Created
              </div>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <div className="text-lg font-semibold text-zinc-900">
                {lastResult.skipped_count}
              </div>
              <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">
                Skipped
              </div>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <div className="text-lg font-semibold text-zinc-900">
                {lastResult.error_count}
              </div>
              <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">
                Errors
              </div>
            </div>
          </div>

          <div className="max-h-64 overflow-y-auto rounded-xl border border-zinc-200 bg-white">
            <div className="flex flex-col divide-y divide-zinc-100">
              {lastResult.results.map((row) => (
                <div
                  key={`${row.row_number}-${row.email ?? row.message}`}
                  className="flex items-start gap-3 px-3 py-3"
                >
                  <Badge variant={getStatusVariant(row.status)} size="small">
                    {row.status}
                  </Badge>
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <span className="text-sm font-medium text-zinc-900">
                      Row {row.row_number}
                      {row.email ? ` · ${row.email}` : ""}
                    </span>
                    <span className="text-xs text-zinc-500">{row.message}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </form>
  );
}
