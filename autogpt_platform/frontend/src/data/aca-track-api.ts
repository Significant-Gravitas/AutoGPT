/**
 * ACA Track API Endpoints
 *
 * Mock API endpoint documentation for prototype development.
 * These endpoints support the ACA Track Figma prototype interactions.
 */

// ============================================
// BASE CONFIGURATION
// ============================================
export const API_BASE_URL = '/api';
export const API_VERSION = 'v1';

// ============================================
// CUSTOMER ENDPOINTS
// ============================================
export const customerEndpoints = {
  /**
   * GET /api/customers
   *
   * List all customers with pagination and filtering
   *
   * Query Parameters:
   * - page: number (default: 1)
   * - per_page: number (default: 50, max: 100)
   * - sort_by: string (column name)
   * - sort_order: 'asc' | 'desc'
   * - pod: string (Pod A, Pod B, Pod C, Pod D)
   * - tier: string (Tier 1, Tier 2, Tier 3)
   * - status: string (filter by filing/onboarding status)
   * - data_quality_min: number (0-100)
   * - data_quality_max: number (0-100)
   * - has_errors: boolean
   * - past_deadline: boolean
   * - search: string (name or EIN)
   *
   * Response: {
   *   data: Customer[],
   *   pagination: {
   *     page: number,
   *     per_page: number,
   *     total: number,
   *     total_pages: number
   *   },
   *   kpis: KPIMetrics
   * }
   */
  list: `${API_BASE_URL}/customers`,

  /**
   * GET /api/customers/:id
   *
   * Get single customer details
   *
   * Response: Customer (full object with expanded relations)
   */
  get: (id: number | string) => `${API_BASE_URL}/customers/${id}`,

  /**
   * GET /api/customers/:id/files
   *
   * Get customer file upload history
   *
   * Query Parameters:
   * - page: number
   * - per_page: number
   * - file_type: string
   *
   * Response: {
   *   data: FileUpload[],
   *   pagination: {...}
   * }
   */
  files: (id: number | string) => `${API_BASE_URL}/customers/${id}/files`,

  /**
   * POST /api/customers/:id/files
   *
   * Upload a new file for customer
   *
   * Body (multipart/form-data):
   * - file: File
   * - upload_method: 'CSV' | 'Portal' | 'SFTP'
   *
   * Response: FileUpload
   */
  uploadFile: (id: number | string) => `${API_BASE_URL}/customers/${id}/files`,

  /**
   * POST /api/customers/:id/notifications
   *
   * Send notification to customer
   *
   * Body: {
   *   template_id: string,
   *   recipient_email: string,
   *   recipient_name: string,
   *   scheduled_at?: string (ISO date),
   *   custom_message?: string
   * }
   *
   * Response: {
   *   notification_id: string,
   *   status: 'queued' | 'sent',
   *   sent_at?: string
   * }
   */
  sendNotification: (id: number | string) =>
    `${API_BASE_URL}/customers/${id}/notifications`,

  /**
   * GET /api/customers/:id/audit-log
   *
   * Get customer audit log
   *
   * Query Parameters:
   * - page: number
   * - per_page: number
   * - action_type: string
   * - actor: string
   * - start_date: string
   * - end_date: string
   *
   * Response: {
   *   data: AuditLogEntry[],
   *   pagination: {...}
   * }
   */
  auditLog: (id: number | string) => `${API_BASE_URL}/customers/${id}/audit-log`,

  /**
   * GET /api/customers/:id/filings
   *
   * Get customer filing history
   *
   * Response: Filing[]
   */
  filings: (id: number | string) => `${API_BASE_URL}/customers/${id}/filings`,

  /**
   * POST /api/customers/:id/filings/:filingId/resubmit
   *
   * Resubmit a corrected filing
   *
   * Body: {
   *   corrections: ValidationError[]
   * }
   *
   * Response: Filing
   */
  resubmitFiling: (customerId: number | string, filingId: string) =>
    `${API_BASE_URL}/customers/${customerId}/filings/${filingId}/resubmit`,
};

// ============================================
// FILE ENDPOINTS
// ============================================
export const fileEndpoints = {
  /**
   * GET /api/files/:id/validation-report
   *
   * Get validation report for a file
   *
   * Response: {
   *   file_id: string,
   *   validation_run_id: string,
   *   timestamp: string,
   *   status: 'Passed' | 'Warnings' | 'Failed',
   *   summary: {
   *     critical: number,
   *     warning: number,
   *     info: number
   *   },
   *   errors: Array<{
   *     code: string,
   *     severity: string,
   *     count: number,
   *     samples: Array<{
   *       row: number,
   *       field: string,
   *       value: string,
   *       expected: string
   *     }>,
   *     suggested_fix: string
   *   }>,
   *   diff?: {
   *     added: number,
   *     removed: number,
   *     modified: number,
   *     changes: Array<{
   *       type: string,
   *       field: string,
   *       old_value: string,
   *       new_value: string
   *     }>
   *   }
   * }
   */
  validationReport: (fileId: string) =>
    `${API_BASE_URL}/files/${fileId}/validation-report`,

  /**
   * POST /api/files/:id/reprocess
   *
   * Trigger reprocessing of a file
   *
   * Response: {
   *   job_id: string,
   *   status: 'queued'
   * }
   */
  reprocess: (fileId: string) => `${API_BASE_URL}/files/${fileId}/reprocess`,

  /**
   * POST /api/files/:id/accept-fixes
   *
   * Accept suggested fixes and queue for manual edit
   *
   * Body: {
   *   fixes: Array<{
   *     error_code: string,
   *     apply: boolean
   *   }>
   * }
   *
   * Response: {
   *   edit_queue_id: string,
   *   fixes_applied: number
   * }
   */
  acceptFixes: (fileId: string) => `${API_BASE_URL}/files/${fileId}/accept-fixes`,

  /**
   * GET /api/files/:id/download
   *
   * Download original file
   *
   * Response: Binary file stream
   */
  download: (fileId: string) => `${API_BASE_URL}/files/${fileId}/download`,
};

// ============================================
// POD ENDPOINTS
// ============================================
export const podEndpoints = {
  /**
   * GET /api/pods
   *
   * List all pods
   *
   * Response: Pod[]
   */
  list: `${API_BASE_URL}/pods`,

  /**
   * GET /api/pods/:id
   *
   * Get pod details
   *
   * Response: Pod (with expanded customer list)
   */
  get: (id: string) => `${API_BASE_URL}/pods/${id}`,

  /**
   * GET /api/pods/:id/sla-metrics
   *
   * Get SLA metrics for a pod
   *
   * Query Parameters:
   * - period: string (YYYY-MM or YYYY)
   * - granularity: 'day' | 'week' | 'month'
   *
   * Response: {
   *   pod_id: string,
   *   period: string,
   *   compliance_rate: number,
   *   avg_response_time: number,
   *   tickets_resolved: number,
   *   tickets_escalated: number,
   *   trend: Array<{
   *     date: string,
   *     compliance: number
   *   }>
   * }
   */
  slaMetrics: (podId: string) => `${API_BASE_URL}/pods/${podId}/sla-metrics`,
};

// ============================================
// PRINTING ENDPOINTS
// ============================================
export const printingEndpoints = {
  /**
   * GET /api/print-manifests
   *
   * List print manifests
   *
   * Query Parameters:
   * - customer_id: number
   * - status: string
   * - start_date: string
   * - end_date: string
   *
   * Response: PrintManifest[]
   */
  list: `${API_BASE_URL}/print-manifests`,

  /**
   * GET /api/print-manifests/:batchId
   *
   * Get print manifest details
   *
   * Response: PrintManifest (with tracking and piece details)
   */
  get: (batchId: string) => `${API_BASE_URL}/print-manifests/${batchId}`,

  /**
   * POST /api/print-manifests/:batchId/reschedule
   *
   * Reschedule a print batch
   *
   * Body: {
   *   scheduled_date: string (ISO date)
   * }
   *
   * Response: PrintManifest
   */
  reschedule: (batchId: string) =>
    `${API_BASE_URL}/print-manifests/${batchId}/reschedule`,

  /**
   * GET /api/print-manifests/:batchId/preview
   *
   * Preview form before printing
   *
   * Response: {
   *   preview_url: string,
   *   pages: number
   * }
   */
  preview: (batchId: string) =>
    `${API_BASE_URL}/print-manifests/${batchId}/preview`,

  /**
   * GET /api/returned-mail
   *
   * List returned mail
   *
   * Query Parameters:
   * - customer_id: number
   * - resolved: boolean
   *
   * Response: Array<{
   *   id: string,
   *   batch_id: string,
   *   original_address: Address,
   *   return_reason: string,
   *   returned_at: string,
   *   resolved: boolean
   * }>
   */
  returnedMail: `${API_BASE_URL}/returned-mail`,

  /**
   * POST /api/returned-mail/:id/reprint
   *
   * Reprint with corrected address
   *
   * Body: {
   *   corrected_address: Address
   * }
   *
   * Response: PrintManifest
   */
  reprint: (id: string) => `${API_BASE_URL}/returned-mail/${id}/reprint`,
};

// ============================================
// SUPPORT TICKET ENDPOINTS
// ============================================
export const ticketEndpoints = {
  /**
   * GET /api/tickets
   *
   * List support tickets
   *
   * Query Parameters:
   * - customer_id: number
   * - status: string
   * - severity: string
   * - assigned_to: string
   *
   * Response: SupportTicket[]
   */
  list: `${API_BASE_URL}/tickets`,

  /**
   * GET /api/tickets/:id
   *
   * Get ticket details with timeline
   *
   * Response: SupportTicket (with full timeline)
   */
  get: (id: number | string) => `${API_BASE_URL}/tickets/${id}`,

  /**
   * POST /api/tickets/:id/escalate
   *
   * Escalate ticket to pod lead
   *
   * Body: {
   *   reason: string,
   *   notify_channels: ('email' | 'slack')[]
   * }
   *
   * Response: {
   *   escalated: true,
   *   notified: string[]
   * }
   */
  escalate: (id: number | string) => `${API_BASE_URL}/tickets/${id}/escalate`,

  /**
   * POST /api/tickets/:id/request-confirmation
   *
   * Request customer confirmation
   *
   * Body: {
   *   message: string
   * }
   *
   * Response: {
   *   confirmation_requested: true,
   *   sent_to: string
   * }
   */
  requestConfirmation: (id: number | string) =>
    `${API_BASE_URL}/tickets/${id}/request-confirmation`,
};

// ============================================
// NOTIFICATION TEMPLATES
// ============================================
export const notificationEndpoints = {
  /**
   * GET /api/notification-templates
   *
   * List available notification templates
   *
   * Response: Array<{
   *   id: string,
   *   name: string,
   *   subject: string,
   *   body_preview: string,
   *   variables: string[]
   * }>
   */
  templates: `${API_BASE_URL}/notification-templates`,

  /**
   * POST /api/notification-templates/:id/preview
   *
   * Preview notification with variables filled
   *
   * Body: {
   *   variables: Record<string, string>
   * }
   *
   * Response: {
   *   subject: string,
   *   body: string
   * }
   */
  preview: (templateId: string) =>
    `${API_BASE_URL}/notification-templates/${templateId}/preview`,
};

// ============================================
// USER / SAVED VIEWS ENDPOINTS
// ============================================
export const userEndpoints = {
  /**
   * GET /api/users/me
   *
   * Get current user profile
   *
   * Response: {
   *   id: string,
   *   name: string,
   *   email: string,
   *   role: string,
   *   pod: string,
   *   preferences: UserPreferences
   * }
   */
  me: `${API_BASE_URL}/users/me`,

  /**
   * GET /api/users/me/saved-views
   *
   * Get user's saved views
   *
   * Response: Array<{
   *   id: string,
   *   name: string,
   *   filters: Record<string, any>,
   *   columns: string[],
   *   sort_by: string,
   *   sort_order: string,
   *   is_default: boolean
   * }>
   */
  savedViews: `${API_BASE_URL}/users/me/saved-views`,

  /**
   * POST /api/users/me/saved-views
   *
   * Create a new saved view
   *
   * Body: SavedView (without id)
   *
   * Response: SavedView
   */
  createSavedView: `${API_BASE_URL}/users/me/saved-views`,

  /**
   * DELETE /api/users/me/saved-views/:id
   *
   * Delete a saved view
   */
  deleteSavedView: (id: string) => `${API_BASE_URL}/users/me/saved-views/${id}`,
};

// ============================================
// KPI / ANALYTICS ENDPOINTS
// ============================================
export const analyticsEndpoints = {
  /**
   * GET /api/analytics/kpis
   *
   * Get overview KPIs
   *
   * Response: KPIMetrics
   */
  kpis: `${API_BASE_URL}/analytics/kpis`,

  /**
   * GET /api/analytics/error-trends
   *
   * Get error rate trends
   *
   * Query Parameters:
   * - period: 'week' | 'month' | 'quarter' | 'year'
   *
   * Response: Array<{
   *   date: string,
   *   error_count: number,
   *   customer_count: number
   * }>
   */
  errorTrends: `${API_BASE_URL}/analytics/error-trends`,

  /**
   * GET /api/analytics/top-errors
   *
   * Get top error types
   *
   * Query Parameters:
   * - limit: number (default: 10)
   *
   * Response: Array<{
   *   code: string,
   *   description: string,
   *   count: number,
   *   affected_customers: number
   * }>
   */
  topErrors: `${API_BASE_URL}/analytics/top-errors`,

  /**
   * GET /api/analytics/pod-utilization
   *
   * Get pod utilization metrics
   *
   * Response: Array<{
   *   pod_id: string,
   *   pod_name: string,
   *   customer_count: number,
   *   utilization_percent: number
   * }>
   */
  podUtilization: `${API_BASE_URL}/analytics/pod-utilization`,

  /**
   * GET /api/analytics/filing-status
   *
   * Get filing acceptance/rejection stats
   *
   * Response: {
   *   accepted: number,
   *   rejected: number,
   *   pending: number,
   *   not_started: number
   * }
   */
  filingStatus: `${API_BASE_URL}/analytics/filing-status`,

  /**
   * GET /api/analytics/penalty-heatmap
   *
   * Get penalty exposure heatmap data
   *
   * Response: Array<{
   *   customer_id: number,
   *   customer_name: string,
   *   penalty_exposure: number,
   *   risk_factors: string[]
   * }>
   */
  penaltyHeatmap: `${API_BASE_URL}/analytics/penalty-heatmap`,
};

// ============================================
// TYPE DEFINITIONS FOR API RESPONSES
// ============================================
export interface APIResponse<T> {
  data: T;
  meta?: {
    request_id: string;
    timestamp: string;
  };
}

export interface PaginatedResponse<T> extends APIResponse<T[]> {
  pagination: {
    page: number;
    per_page: number;
    total: number;
    total_pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, string[]>;
  };
}

// ============================================
// EXAMPLE MOCK DATA FETCHER
// ============================================
export async function mockFetch<T>(
  endpoint: string,
  _options?: RequestInit
): Promise<APIResponse<T>> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 200 + Math.random() * 300));

  // In a real implementation, this would make actual API calls
  // For the prototype, return mock data based on endpoint

  return {
    data: {} as T,
    meta: {
      request_id: `req_${Date.now()}`,
      timestamp: new Date().toISOString(),
    },
  };
}
