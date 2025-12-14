/**
 * ACA Track Data Types
 *
 * TypeScript interfaces for the ACA Track prototype data model
 */

export interface ValidationError {
  code: string;
  severity: 'Critical' | 'High' | 'Medium' | 'Warning' | 'Low';
  count: number;
  sample: string;
}

export interface SupportTicket {
  id: number;
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  status: 'Open' | 'In Progress' | 'Resolved' | 'Closed';
  assignedTo?: string;
  slaRemaining?: string;
  timeline?: TicketEvent[];
}

export interface TicketEvent {
  timestamp: string;
  actor: string;
  action: string;
  notes?: string;
  attachments?: string[];
}

export type OnboardingStatus = 'Not started' | 'Partially onboarded' | 'Onboarded';
export type FileType = 'CSV' | 'SFTP' | 'Portal' | 'API';
export type FilingStatus = 'Not started' | 'Draft' | 'Submitted' | 'Accepted' | 'Rejected' | 'Not filed' | 'Pending';
export type PrintingStatus = 'Not scheduled' | 'Scheduled' | 'In progress' | 'Printed' | 'Mailed';
export type ContractTier = 'Tier 1' | 'Tier 2' | 'Tier 3';
export type ExecutivePod = 'Pod A' | 'Pod B' | 'Pod C' | 'Pod D';

export interface Customer {
  customer_id: number;
  customer_name: string;
  EINs: string;
  primary_contact_name: string;
  primary_contact_email: string;
  executive_pod: ExecutivePod;
  contract_tier: ContractTier;
  PEPM_rate: number;
  per_form_rate: number;
  num_eins: number;
  num_forms_last_year: number;
  onboarding_status: OnboardingStatus;
  last_file_upload_timestamp: string;
  last_file_type: FileType;
  last_plan_update_timestamp: string;
  deadline_date: string;
  data_quality_score: number;
  penalty_exposure: number;
  printing_status: PrintingStatus;
  mail_batch_id: string;
  '1094_status': FilingStatus;
  '1095_status': FilingStatus;
  filing_transaction_ids: string[];
  validation_errors_json: ValidationError[];
  support_tickets_json: SupportTicket[];
  last_sla_review: string;
}

// Row state determination for table styling
export type RowState = 'normal' | 'warning' | 'critical' | 'verified';

export function getRowState(customer: Customer): RowState {
  if (customer.data_quality_score >= 90 && customer.validation_errors_json.length === 0) {
    return 'verified';
  }
  const hasCriticalErrors = customer.validation_errors_json.some(
    (e) => e.severity === 'Critical'
  );
  if (hasCriticalErrors || customer.data_quality_score < 50) {
    return 'critical';
  }
  const isAtRisk =
    customer.validation_errors_json.length > 0 ||
    customer.data_quality_score < 70 ||
    new Date(customer.deadline_date) < new Date();
  if (isAtRisk) {
    return 'warning';
  }
  return 'normal';
}

// KPI calculations
export interface KPIMetrics {
  totalCustomers: number;
  cleanAndVerified: number;
  customersWithErrors: number;
  pastDeadline: number;
  pastDeadlineDays: number;
  estimatedPenaltyExposure: number;
  slaCompliance30d: number;
  slaCompliance90d: number;
}

export function calculateKPIs(customers: Customer[]): KPIMetrics {
  const today = new Date();

  const cleanAndVerified = customers.filter(
    (c) => c.data_quality_score >= 90 && c.validation_errors_json.length === 0
  ).length;

  const customersWithErrors = customers.filter(
    (c) => c.validation_errors_json.length > 0
  ).length;

  const pastDeadlineCustomers = customers.filter(
    (c) => new Date(c.deadline_date) < today && c['1094_status'] !== 'Submitted' && c['1094_status'] !== 'Accepted'
  );

  const pastDeadlineDays = pastDeadlineCustomers.reduce((total, c) => {
    const deadline = new Date(c.deadline_date);
    const daysPast = Math.floor((today.getTime() - deadline.getTime()) / (1000 * 60 * 60 * 24));
    return total + daysPast;
  }, 0);

  const estimatedPenaltyExposure = customers.reduce(
    (total, c) => total + c.penalty_exposure,
    0
  );

  // Mock SLA compliance (would be calculated from actual SLA data)
  const slaCompliance30d = 94.5;
  const slaCompliance90d = 92.3;

  return {
    totalCustomers: customers.length,
    cleanAndVerified,
    customersWithErrors,
    pastDeadline: pastDeadlineCustomers.length,
    pastDeadlineDays,
    estimatedPenaltyExposure,
    slaCompliance30d,
    slaCompliance90d,
  };
}

// File upload record
export interface FileUpload {
  id: string;
  fileName: string;
  uploadedBy: string;
  uploadMethod: FileType;
  timestamp: string;
  size: string;
  processingStatus: 'Pending' | 'Processing' | 'Completed' | 'Failed';
  validationRunId?: string;
}

// Validation run record
export interface ValidationRun {
  id: string;
  fileId: string;
  timestamp: string;
  status: 'Passed' | 'Warnings' | 'Failed';
  criticalCount: number;
  warningCount: number;
  infoCount: number;
  errors: ValidationError[];
}

// Filing record
export interface Filing {
  id: string;
  formType: '1094-C' | '1095-C';
  status: FilingStatus;
  irsTransactionId?: string;
  irsResponseCode?: string;
  irsResponseMessage?: string;
  submittedAt?: string;
  acceptedAt?: string;
  rejectedAt?: string;
  rejectionDetails?: string[];
}

// Print manifest record
export interface PrintManifest {
  batchId: string;
  status: PrintingStatus;
  pieces: number;
  postage: number;
  tracking?: string;
  scheduledDate?: string;
  printedDate?: string;
  mailedDate?: string;
}

// Audit log entry
export interface AuditLogEntry {
  id: string;
  timestamp: string;
  actor: string;
  actorType: 'User' | 'System' | 'API';
  action: string;
  actionType: 'upload' | 'edit' | 'notification' | 'filing' | 'print' | 'validation' | 'login' | 'other';
  details?: string;
  resourceType?: string;
  resourceId?: string;
}
