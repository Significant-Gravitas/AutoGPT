"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

type ModelMode = "standard" | "max" | "auto";
type AgentPersona =
  | "frontend_dev"
  | "backend_dev"
  | "fullstack_dev"
  | "devops"
  | "security_auditor"
  | "data_engineer"
  | "code_reviewer"
  | "documentation_writer"
  | "test_engineer"
  | "custom";

type TaskStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

interface Task {
  id: string;
  prompt: string;
  priority: number;
  persona: AgentPersona;
  model_mode: ModelMode;
  status: TaskStatus;
  created_at: string;
  tags: string[];
}

interface JournalEntry {
  id: string;
  title: string;
  summary: string;
  outcome: "success" | "failure" | "partial" | "cancelled";
  model_mode: string;
  persona: string;
  tokens_used: number;
  execution_time_secs: number;
  files_changed: string[];
  commit_hash: string;
  tags: string[];
  timestamp: string;
}

interface OllamaModel {
  name: string;
  size: string;
  modified: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const PERSONA_LABELS: Record<AgentPersona, string> = {
  frontend_dev: "Frontend Dev",
  backend_dev: "Backend Dev",
  fullstack_dev: "Full Stack Dev",
  devops: "DevOps",
  security_auditor: "Security Auditor",
  data_engineer: "Data Engineer",
  code_reviewer: "Code Reviewer",
  documentation_writer: "Docs Writer",
  test_engineer: "Test Engineer",
  custom: "Custom",
};

const PERSONA_COLORS: Record<AgentPersona, string> = {
  frontend_dev: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  backend_dev: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  fullstack_dev: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  devops: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  security_auditor: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  data_engineer: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  code_reviewer: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200",
  documentation_writer: "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200",
  test_engineer: "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200",
  custom: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
};

const STATUS_COLORS: Record<TaskStatus, string> = {
  pending: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  cancelled: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
};

const OUTCOME_ICONS: Record<string, string> = {
  success: "✅",
  failure: "❌",
  partial: "⚠️",
  cancelled: "🚫",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function ModelModeToggle({
  mode,
  onChange,
}: {
  mode: ModelMode;
  onChange: (m: ModelMode) => void;
}) {
  const options: { value: ModelMode; label: string; desc: string }[] = [
    { value: "standard", label: "Standard", desc: "Local Ollama (fast, private)" },
    { value: "auto", label: "Auto", desc: "Smart routing by complexity" },
    { value: "max", label: "Max", desc: "NVIDIA NIM (powerful, cloud)" },
  ];
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
        Model Mode
      </label>
      <div className="flex gap-1 rounded-lg bg-gray-100 p-1 dark:bg-gray-800">
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            title={opt.desc}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-all ${
              mode === opt.value
                ? "bg-white text-gray-900 shadow-sm dark:bg-gray-700 dark:text-white"
                : "text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function PersonaSelector({
  persona,
  onChange,
}: {
  persona: AgentPersona;
  onChange: (p: AgentPersona) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
        Agent Persona
      </label>
      <select
        value={persona}
        onChange={(e) => onChange(e.target.value as AgentPersona)}
        className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
      >
        {Object.entries(PERSONA_LABELS).map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
    </div>
  );
}

function TaskCard({
  task,
  onRemove,
}: {
  task: Task;
  onRemove: (id: string) => void;
}) {
  return (
    <div className="group flex items-start gap-3 rounded-lg border border-gray-200 bg-white p-3 shadow-sm transition-all hover:border-blue-300 dark:border-gray-700 dark:bg-gray-800">
      <div className="flex-1 min-w-0">
        <p className="truncate text-sm text-gray-900 dark:text-white">
          {task.prompt}
        </p>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          <span
            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${PERSONA_COLORS[task.persona]}`}
          >
            {PERSONA_LABELS[task.persona]}
          </span>
          <span
            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_COLORS[task.status]}`}
          >
            {task.status}
          </span>
          <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600 dark:bg-gray-700 dark:text-gray-300">
            {task.model_mode === "max" ? "⚡ Max" : task.model_mode === "standard" ? "🏠 Standard" : "🔀 Auto"}
          </span>
          {task.tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500 dark:bg-gray-700 dark:text-gray-400"
            >
              #{tag}
            </span>
          ))}
        </div>
      </div>
      <button
        onClick={() => onRemove(task.id)}
        className="opacity-0 group-hover:opacity-100 rounded p-1 text-gray-400 hover:text-red-500 transition-all"
        title="Remove task"
      >
        ✕
      </button>
    </div>
  );
}

function JournalEntryCard({ entry }: { entry: JournalEntry }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-700 dark:bg-gray-800">
      <div
        className="flex cursor-pointer items-start justify-between gap-2"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-base">{OUTCOME_ICONS[entry.outcome] || "📝"}</span>
            <p className="truncate text-sm font-medium text-gray-900 dark:text-white">
              {entry.title}
            </p>
          </div>
          <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
            {new Date(entry.timestamp).toLocaleString()} •{" "}
            {entry.persona?.replace(/_/g, " ")} •{" "}
            {entry.model_mode || "auto"}
          </p>
        </div>
        <span className="text-gray-400 text-xs">{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="mt-3 space-y-2 border-t border-gray-100 pt-3 dark:border-gray-700">
          {entry.summary && (
            <p className="text-sm text-gray-700 dark:text-gray-300">{entry.summary}</p>
          )}
          <div className="flex flex-wrap gap-3 text-xs text-gray-500 dark:text-gray-400">
            {entry.tokens_used > 0 && <span>🔢 {entry.tokens_used.toLocaleString()} tokens</span>}
            {entry.execution_time_secs > 0 && <span>⏱ {entry.execution_time_secs}s</span>}
            {entry.commit_hash && <span>📌 {entry.commit_hash}</span>}
          </div>
          {entry.files_changed?.length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400">
                Files changed ({entry.files_changed.length}):
              </p>
              <ul className="mt-1 space-y-0.5">
                {entry.files_changed.slice(0, 5).map((f) => (
                  <li key={f} className="text-xs font-mono text-gray-600 dark:text-gray-300">
                    • {f}
                  </li>
                ))}
                {entry.files_changed.length > 5 && (
                  <li className="text-xs text-gray-400">
                    ...and {entry.files_changed.length - 5} more
                  </li>
                )}
              </ul>
            </div>
          )}
          {entry.tags?.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {entry.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500 dark:bg-gray-700 dark:text-gray-400"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function OllamaManager({
  models,
  selectedModel,
  onSelectModel,
  onRefresh,
  isLoading,
}: {
  models: OllamaModel[];
  selectedModel: string;
  onSelectModel: (m: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
          Ollama Models
        </label>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="rounded p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 disabled:opacity-50"
          title="Refresh models"
        >
          {isLoading ? "⟳" : "↻"}
        </button>
      </div>
      {models.length === 0 ? (
        <p className="text-xs text-gray-400 dark:text-gray-500">
          No models found. Ensure Ollama is running.
        </p>
      ) : (
        <div className="space-y-1">
          {models.map((m) => (
            <button
              key={m.name}
              onClick={() => onSelectModel(m.name)}
              className={`w-full rounded-lg px-3 py-2 text-left text-sm transition-all ${
                selectedModel === m.name
                  ? "bg-blue-50 text-blue-700 border border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800"
                  : "text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700"
              }`}
            >
              <span className="font-medium">{m.name}</span>
              {m.size && (
                <span className="ml-2 text-xs text-gray-400">{m.size}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────

export function CodingAgentDashboard() {
  // State
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState<"queue" | "journal" | "ollama" | "settings">("queue");
  const [modelMode, setModelMode] = useState<ModelMode>("auto");
  const [persona, setPersona] = useState<AgentPersona>("fullstack_dev");
  const [taskInput, setTaskInput] = useState("");
  const [priority, setPriority] = useState(5);
  const [tags, setTags] = useState("");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);
  const [journalSearch, setJournalSearch] = useState("");
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState("llama3.3");
  const [ollamaLoading, setOllamaLoading] = useState(false);
  const [ollamaHost, setOllamaHost] = useState("http://localhost:11434");
  const [discordWebhook, setDiscordWebhook] = useState("");
  const [slackWebhook, setSlackWebhook] = useState("");
  const [autoCommit, setAutoCommit] = useState(true);
  const [notification, setNotification] = useState<{ msg: string; type: "success" | "error" | "info" } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Dark mode effect
  useEffect(() => {
    const saved = localStorage.getItem("coding-agent-dark-mode");
    if (saved === "true") setDarkMode(true);
  }, []);

  useEffect(() => {
    localStorage.setItem("coding-agent-dark-mode", String(darkMode));
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  // Load persisted state
  useEffect(() => {
    const savedTasks = localStorage.getItem("coding-agent-tasks");
    if (savedTasks) {
      try { setTasks(JSON.parse(savedTasks)); } catch {}
    }
    const savedJournal = localStorage.getItem("coding-agent-journal");
    if (savedJournal) {
      try { setJournalEntries(JSON.parse(savedJournal)); } catch {}
    }
    const savedSettings = localStorage.getItem("coding-agent-settings");
    if (savedSettings) {
      try {
        const s = JSON.parse(savedSettings);
        if (s.ollamaHost) setOllamaHost(s.ollamaHost);
        if (s.discordWebhook) setDiscordWebhook(s.discordWebhook);
        if (s.slackWebhook) setSlackWebhook(s.slackWebhook);
        if (s.autoCommit !== undefined) setAutoCommit(s.autoCommit);
        if (s.selectedOllamaModel) setSelectedOllamaModel(s.selectedOllamaModel);
      } catch {}
    }
  }, []);

  // Persist tasks
  useEffect(() => {
    localStorage.setItem("coding-agent-tasks", JSON.stringify(tasks));
  }, [tasks]);

  // Persist journal
  useEffect(() => {
    localStorage.setItem("coding-agent-journal", JSON.stringify(journalEntries));
  }, [journalEntries]);

  // Keyboard shortcut: Ctrl+Enter to enqueue
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleEnqueue();
      }
      // Ctrl+K to focus task input
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  // Notification helper
  const showNotification = (msg: string, type: "success" | "error" | "info" = "info") => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // Enqueue task
  const handleEnqueue = useCallback(() => {
    if (!taskInput.trim()) return;
    const newTask: Task = {
      id: Math.random().toString(36).slice(2, 10),
      prompt: taskInput.trim(),
      priority,
      persona,
      model_mode: modelMode,
      status: "pending",
      created_at: new Date().toISOString(),
      tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
    };
    setTasks((prev) => {
      const updated = [...prev, newTask].sort((a, b) => a.priority - b.priority);
      return updated;
    });
    setTaskInput("");
    setTags("");
    showNotification("Task enqueued!", "success");
  }, [taskInput, priority, persona, modelMode, tags]);

  // Remove task
  const handleRemoveTask = (id: string) => {
    setTasks((prev) => prev.filter((t) => t.id !== id));
  };

  // Simulate task completion (demo)
  const handleRunNext = () => {
    const pending = tasks.find((t) => t.status === "pending");
    if (!pending) {
      showNotification("No pending tasks.", "info");
      return;
    }
    setTasks((prev) =>
      prev.map((t) => (t.id === pending.id ? { ...t, status: "running" } : t))
    );
    // Simulate completion after 2s
    setTimeout(() => {
      setTasks((prev) =>
        prev.map((t) => (t.id === pending.id ? { ...t, status: "completed" } : t))
      );
      const entry: JournalEntry = {
        id: Math.random().toString(36).slice(2, 10),
        title: pending.prompt.slice(0, 60),
        summary: `Completed: ${pending.prompt}`,
        outcome: "success",
        model_mode: pending.model_mode,
        persona: pending.persona,
        tokens_used: Math.floor(Math.random() * 5000) + 500,
        execution_time_secs: Math.floor(Math.random() * 120) + 10,
        files_changed: [],
        commit_hash: Math.random().toString(36).slice(2, 9),
        tags: pending.tags,
        timestamp: new Date().toISOString(),
      };
      setJournalEntries((prev) => [entry, ...prev]);
      showNotification("Task completed!", "success");
    }, 2000);
  };

  // Fetch Ollama models
  const fetchOllamaModels = useCallback(async () => {
    setOllamaLoading(true);
    try {
      const resp = await fetch(`${ollamaHost}/api/tags`);
      if (resp.ok) {
        const data = await resp.json();
        const models: OllamaModel[] = (data.models || []).map((m: any) => ({
          name: m.name,
          size: m.size ? `${(m.size / 1e9).toFixed(1)}GB` : "",
          modified: m.modified_at || "",
        }));
        setOllamaModels(models);
        showNotification(`Found ${models.length} Ollama models.`, "success");
      } else {
        showNotification("Failed to connect to Ollama.", "error");
      }
    } catch {
      showNotification("Cannot reach Ollama. Check host URL.", "error");
    } finally {
      setOllamaLoading(false);
    }
  }, [ollamaHost]);

  // Save settings
  const saveSettings = () => {
    localStorage.setItem(
      "coding-agent-settings",
      JSON.stringify({ ollamaHost, discordWebhook, slackWebhook, autoCommit, selectedOllamaModel })
    );
    showNotification("Settings saved.", "success");
  };

  // Filter journal
  const filteredJournal = journalSearch
    ? journalEntries.filter(
        (e) =>
          e.title.toLowerCase().includes(journalSearch.toLowerCase()) ||
          e.summary.toLowerCase().includes(journalSearch.toLowerCase()) ||
          e.tags.join(" ").toLowerCase().includes(journalSearch.toLowerCase())
      )
    : journalEntries;

  const pendingCount = tasks.filter((t) => t.status === "pending").length;
  const runningCount = tasks.filter((t) => t.status === "running").length;

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-950 ${darkMode ? "dark" : ""}`}>
      {/* Notification Toast */}
      {notification && (
        <div
          className={`fixed right-4 top-4 z-50 rounded-lg px-4 py-3 text-sm font-medium shadow-lg transition-all ${
            notification.type === "success"
              ? "bg-green-500 text-white"
              : notification.type === "error"
              ? "bg-red-500 text-white"
              : "bg-blue-500 text-white"
          }`}
        >
          {notification.msg}
        </div>
      )}

      <div className="mx-auto max-w-7xl px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              🤖 Coding Agent
            </h1>
            <p className="mt-0.5 text-sm text-gray-500 dark:text-gray-400">
              Personal autonomous coding assistant with NVIDIA NIM + Ollama
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Stats */}
            <div className="flex gap-2">
              {pendingCount > 0 && (
                <span className="rounded-full bg-yellow-100 px-3 py-1 text-xs font-medium text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                  {pendingCount} pending
                </span>
              )}
              {runningCount > 0 && (
                <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-200 animate-pulse">
                  {runningCount} running
                </span>
              )}
            </div>
            {/* Dark mode toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
              title="Toggle dark/light mode"
            >
              {darkMode ? "☀️ Light" : "🌙 Dark"}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Left panel: Task input */}
          <div className="lg:col-span-2 space-y-4">
            {/* Task input card */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900">
              <h2 className="mb-4 text-base font-semibold text-gray-900 dark:text-white">
                New Task
              </h2>
              <div className="space-y-4">
                <div>
                  <textarea
                    ref={textareaRef}
                    value={taskInput}
                    onChange={(e) => setTaskInput(e.target.value)}
                    placeholder="Describe the coding task... (Ctrl+Enter to enqueue, Ctrl+K to focus)"
                    rows={4}
                    className="w-full rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white dark:placeholder-gray-500 dark:focus:bg-gray-800"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <ModelModeToggle mode={modelMode} onChange={setModelMode} />
                  <PersonaSelector persona={persona} onChange={setPersona} />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                      Priority (1=high, 10=low)
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={priority}
                      onChange={(e) => setPriority(Number(e.target.value))}
                      className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                      Tags (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={tags}
                      onChange={(e) => setTags(e.target.value)}
                      placeholder="frontend, bugfix, urgent"
                      className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white dark:placeholder-gray-500"
                    />
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={handleEnqueue}
                    disabled={!taskInput.trim()}
                    className="flex-1 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    + Enqueue Task
                  </button>
                  <button
                    onClick={handleRunNext}
                    disabled={pendingCount === 0}
                    className="rounded-lg border border-gray-200 bg-white px-4 py-2.5 text-sm font-semibold text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
                  >
                    ▶ Run Next
                  </button>
                </div>
              </div>
            </div>

            {/* Tabs */}
            <div className="rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-900">
              <div className="flex border-b border-gray-200 dark:border-gray-700">
                {(["queue", "journal", "ollama", "settings"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`flex-1 px-4 py-3 text-sm font-medium capitalize transition-colors ${
                      activeTab === tab
                        ? "border-b-2 border-blue-600 text-blue-600 dark:text-blue-400"
                        : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    }`}
                  >
                    {tab === "queue" && `📋 Queue (${pendingCount})`}
                    {tab === "journal" && `📔 Journal (${journalEntries.length})`}
                    {tab === "ollama" && "🦙 Ollama"}
                    {tab === "settings" && "⚙️ Settings"}
                  </button>
                ))}
              </div>

              <div className="p-4">
                {/* Task Queue Tab */}
                {activeTab === "queue" && (
                  <div className="space-y-2">
                    {tasks.length === 0 ? (
                      <div className="py-8 text-center">
                        <p className="text-gray-400 dark:text-gray-500">
                          No tasks in queue. Add a task above to get started.
                        </p>
                        <p className="mt-1 text-xs text-gray-300 dark:text-gray-600">
                          Tip: Use Ctrl+Enter to quickly enqueue
                        </p>
                      </div>
                    ) : (
                      <>
                        <div className="mb-2 flex items-center justify-between">
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {tasks.length} task{tasks.length !== 1 ? "s" : ""} •{" "}
                            {pendingCount} pending • {runningCount} running •{" "}
                            {tasks.filter((t) => t.status === "completed").length} completed
                          </p>
                          <button
                            onClick={() => setTasks([])}
                            className="text-xs text-red-400 hover:text-red-600"
                          >
                            Clear all
                          </button>
                        </div>
                        {tasks.map((task) => (
                          <TaskCard
                            key={task.id}
                            task={task}
                            onRemove={handleRemoveTask}
                          />
                        ))}
                      </>
                    )}
                  </div>
                )}

                {/* Journal Tab */}
                {activeTab === "journal" && (
                  <div className="space-y-3">
                    <input
                      type="text"
                      value={journalSearch}
                      onChange={(e) => setJournalSearch(e.target.value)}
                      placeholder="Search journal entries..."
                      className="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white dark:placeholder-gray-500"
                    />
                    {filteredJournal.length === 0 ? (
                      <div className="py-8 text-center">
                        <p className="text-gray-400 dark:text-gray-500">
                          {journalSearch ? "No entries match your search." : "No journal entries yet."}
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {filteredJournal.map((entry) => (
                          <JournalEntryCard key={entry.id} entry={entry} />
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Ollama Tab */}
                {activeTab === "ollama" && (
                  <div className="space-y-4">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={ollamaHost}
                        onChange={(e) => setOllamaHost(e.target.value)}
                        placeholder="http://localhost:11434"
                        className="flex-1 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                      />
                      <button
                        onClick={fetchOllamaModels}
                        disabled={ollamaLoading}
                        className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                      >
                        Connect
                      </button>
                    </div>
                    <OllamaManager
                      models={ollamaModels}
                      selectedModel={selectedOllamaModel}
                      onSelectModel={setSelectedOllamaModel}
                      onRefresh={fetchOllamaModels}
                      isLoading={ollamaLoading}
                    />
                    {selectedOllamaModel && (
                      <div className="rounded-lg bg-blue-50 p-3 text-sm text-blue-700 dark:bg-blue-900/20 dark:text-blue-300">
                        Active model: <strong>{selectedOllamaModel}</strong>
                      </div>
                    )}
                  </div>
                )}

                {/* Settings Tab */}
                {activeTab === "settings" && (
                  <div className="space-y-4">
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Notifications
                      </h3>
                      <div>
                        <label className="mb-1 block text-xs text-gray-500 dark:text-gray-400">
                          Discord Webhook URL
                        </label>
                        <input
                          type="url"
                          value={discordWebhook}
                          onChange={(e) => setDiscordWebhook(e.target.value)}
                          placeholder="https://discord.com/api/webhooks/..."
                          className="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-gray-500 dark:text-gray-400">
                          Slack Webhook URL
                        </label>
                        <input
                          type="url"
                          value={slackWebhook}
                          onChange={(e) => setSlackWebhook(e.target.value)}
                          placeholder="https://hooks.slack.com/services/..."
                          className="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                        />
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Git Integration
                      </h3>
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={autoCommit}
                          onChange={(e) => setAutoCommit(e.target.checked)}
                          className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700 dark:text-gray-300">
                          Auto-commit changes with AI-generated commit messages
                        </span>
                      </label>
                    </div>

                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Keyboard Shortcuts
                      </h3>
                      <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
                        <table className="w-full text-xs">
                          <tbody className="space-y-1">
                            {[
                              ["Ctrl+Enter", "Enqueue task"],
                              ["Ctrl+K", "Focus task input"],
                            ].map(([key, desc]) => (
                              <tr key={key}>
                                <td className="py-1 pr-4">
                                  <kbd className="rounded bg-gray-200 px-1.5 py-0.5 font-mono text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                                    {key}
                                  </kbd>
                                </td>
                                <td className="py-1 text-gray-600 dark:text-gray-400">{desc}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <button
                      onClick={saveSettings}
                      className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 transition-colors"
                    >
                      Save Settings
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right panel: Quick stats & model info */}
          <div className="space-y-4">
            {/* Model status card */}
            <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-900">
              <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                Active Configuration
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Mode</span>
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                      modelMode === "max"
                        ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                        : modelMode === "standard"
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                    }`}
                  >
                    {modelMode === "max" ? "⚡ Max (NIM)" : modelMode === "standard" ? "🏠 Standard" : "🔀 Auto"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Persona</span>
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${PERSONA_COLORS[persona]}`}
                  >
                    {PERSONA_LABELS[persona]}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Ollama Model</span>
                  <span className="text-xs font-mono text-gray-700 dark:text-gray-300">
                    {selectedOllamaModel || "not set"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Auto-commit</span>
                  <span
                    className={`text-xs font-medium ${autoCommit ? "text-green-600 dark:text-green-400" : "text-gray-400"}`}
                  >
                    {autoCommit ? "✓ Enabled" : "✗ Disabled"}
                  </span>
                </div>
              </div>
            </div>

            {/* Quick stats */}
            <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-900">
              <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                Session Stats
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Total Tasks", value: tasks.length },
                  { label: "Completed", value: tasks.filter((t) => t.status === "completed").length },
                  { label: "Journal Entries", value: journalEntries.length },
                  {
                    label: "Success Rate",
                    value:
                      journalEntries.length > 0
                        ? `${Math.round((journalEntries.filter((e) => e.outcome === "success").length / journalEntries.length) * 100)}%`
                        : "—",
                  },
                ].map(({ label, value }) => (
                  <div
                    key={label}
                    className="rounded-lg bg-gray-50 p-3 text-center dark:bg-gray-800"
                  >
                    <p className="text-xl font-bold text-gray-900 dark:text-white">{value}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Built-in templates quick access */}
            <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-900">
              <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                Quick Templates
              </h3>
              <div className="space-y-1.5">
                {[
                  { label: "🐛 Fix Bug", prompt: "Fix the following bug:\n\nBug: ", persona: "backend_dev" as AgentPersona },
                  { label: "✨ Add Feature", prompt: "Add the following feature:\n\nFeature: ", persona: "fullstack_dev" as AgentPersona },
                  { label: "🔍 Code Review", prompt: "Review the following code for quality, security, and best practices:\n\n```\n\n```", persona: "code_reviewer" as AgentPersona },
                  { label: "🧪 Write Tests", prompt: "Generate a comprehensive test suite for:\n\n", persona: "test_engineer" as AgentPersona },
                  { label: "📚 Write Docs", prompt: "Generate documentation for:\n\n", persona: "documentation_writer" as AgentPersona },
                  { label: "🔒 Security Audit", prompt: "Perform a security audit of:\n\n", persona: "security_auditor" as AgentPersona },
                ].map(({ label, prompt, persona: p }) => (
                  <button
                    key={label}
                    onClick={() => {
                      setTaskInput(prompt);
                      setPersona(p);
                      textareaRef.current?.focus();
                    }}
                    className="w-full rounded-lg px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-800 transition-colors"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
