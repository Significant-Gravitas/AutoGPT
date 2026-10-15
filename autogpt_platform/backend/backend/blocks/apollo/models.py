from enum import Enum
from typing import Any

from pydantic import BaseModel as OriginalBaseModel
from pydantic import ConfigDict

from backend.data.model import SchemaField


class BaseModel(OriginalBaseModel):
    def model_dump(self, *args, exclude: set[str] | None = None, **kwargs):
        if exclude is None:
            exclude = set("credentials")
        else:
            exclude.add("credentials")

        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump(*args, exclude=exclude, **kwargs)


class PrimaryPhone(BaseModel):
    """A primary phone in Apollo"""

    number: str | None = ""
    source: str | None = ""
    sanitized_number: str | None = ""


class SenorityLevels(str, Enum):
    """Seniority levels in Apollo"""

    OWNER = "owner"
    FOUNDER = "founder"
    C_SUITE = "c_suite"
    PARTNER = "partner"
    VP = "vp"
    HEAD = "head"
    DIRECTOR = "director"
    MANAGER = "manager"
    SENIOR = "senior"
    ENTRY = "entry"
    INTERN = "intern"


class ContactEmailStatuses(str, Enum):
    """Contact email statuses in Apollo"""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    LIKELY_TO_ENGAGE = "likely_to_engage"
    UNAVAILABLE = "unavailable"


class RuleConfigStatus(BaseModel):
    """A rule config status in Apollo"""

    _id: str | None = ""
    created_at: str | None = ""
    rule_action_config_id: str | None = ""
    rule_config_id: str | None = ""
    status_cd: str | None = ""
    updated_at: str | None = ""
    id: str | None = ""
    key: str | None = ""


class ContactCampaignStatus(BaseModel):
    """A contact campaign status in Apollo"""

    id: str | None = ""
    emailer_campaign_id: str | None = ""
    send_email_from_user_id: str | None = ""
    inactive_reason: str | None = ""
    status: str | None = ""
    added_at: str | None = ""
    added_by_user_id: str | None = ""
    finished_at: str | None = ""
    paused_at: str | None = ""
    auto_unpause_at: str | None = ""
    send_email_from_email_address: str | None = ""
    send_email_from_email_account_id: str | None = ""
    manually_set_unpause: str | None = ""
    failure_reason: str | None = ""
    current_step_id: str | None = ""
    in_response_to_emailer_message_id: str | None = ""
    cc_emails: str | None = ""
    bcc_emails: str | None = ""
    to_emails: str | None = ""


class Account(BaseModel):
    """An account in Apollo"""

    id: str | None = ""
    name: str | None = ""
    website_url: str | None = ""
    blog_url: str | None = ""
    angellist_url: str | None = ""
    linkedin_url: str | None = ""
    twitter_url: str | None = ""
    facebook_url: str | None = ""
    primary_phone: PrimaryPhone | None = PrimaryPhone()
    languages: list[str] | None = []
    alexa_ranking: int | None = 0
    phone: str | None = ""
    linkedin_uid: str | None = ""
    founded_year: int | None = 0
    publicly_traded_symbol: str | None = ""
    publicly_traded_exchange: str | None = ""
    logo_url: str | None = ""
    chrunchbase_url: str | None = ""
    primary_domain: str | None = ""
    domain: str | None = ""
    team_id: str | None = ""
    organization_id: str | None = ""
    account_stage_id: str | None = ""
    source: str | None = ""
    original_source: str | None = ""
    creator_id: str | None = ""
    owner_id: str | None = ""
    created_at: str | None = ""
    phone_status: str | None = ""
    hubspot_id: str | None = ""
    salesforce_id: str | None = ""
    crm_owner_id: str | None = ""
    parent_account_id: str | None = ""
    sanitized_phone: str | None = ""
    # no listed type on the API docs
    account_playbook_statues: list[Any] | None = []
    account_rule_config_statuses: list[RuleConfigStatus] | None = []
    existence_level: str | None = ""
    label_ids: list[str] | None = []
    typed_custom_fields: Any | None = {}
    custom_field_errors: Any | None = {}
    modality: str | None = ""
    source_display_name: str | None = ""
    salesforce_record_id: str | None = ""
    crm_record_url: str | None = ""


class ContactEmail(BaseModel):
    """A contact email in Apollo"""

    email: str | None = ""
    email_md5: str | None = ""
    email_sha256: str | None = ""
    email_status: str | None = ""
    email_source: str | None = ""
    extrapolated_email_confidence: str | None = ""
    position: int | None = 0
    email_from_customer: str | None = ""
    free_domain: bool | None = True


class EmploymentHistory(BaseModel):
    """An employment history in Apollo"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    _id: str | None = ""
    created_at: str | None = ""
    current: bool | None = False
    degree: str | None = ""
    description: str | None = ""
    emails: str | None = ""
    end_date: str | None = ""
    grade_level: str | None = ""
    kind: str | None = ""
    major: str | None = ""
    organization_id: str | None = ""
    organization_name: str | None = ""
    raw_address: str | None = ""
    start_date: str | None = ""
    title: str | None = ""
    updated_at: str | None = ""
    id: str | None = ""
    key: str | None = ""


class Breadcrumb(BaseModel):
    """A breadcrumb in Apollo"""

    label: str | None = ""
    signal_field_name: str | None = ""
    value: str | list | None = ""
    display_name: str | None = ""


class TypedCustomField(BaseModel):
    """A typed custom field in Apollo"""

    id: str | None = ""
    value: str | None = ""


class Pagination(BaseModel):
    """Pagination in Apollo"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    page: int = 0
    per_page: int = 0
    total_entries: int = 0
    total_pages: int = 0


class DialerFlags(BaseModel):
    """A dialer flags in Apollo"""

    country_name: str | None = ""
    country_enabled: bool | None = True
    high_risk_calling_enabled: bool | None = True
    potential_high_risk_number: bool | None = True


class PhoneNumber(BaseModel):
    """A phone number in Apollo"""

    raw_number: str | None = ""
    sanitized_number: str | None = ""
    type: str | None = ""
    position: int | None = 0
    status: str | None = ""
    dnc_status: str | None = ""
    dnc_other_info: str | None = ""
    dailer_flags: DialerFlags | None = DialerFlags(
        country_name="",
        country_enabled=True,
        high_risk_calling_enabled=True,
        potential_high_risk_number=True,
    )


class Organization(BaseModel):
    """An organization in Apollo"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    id: str | None = ""
    name: str | None = ""
    website_url: str | None = ""
    blog_url: str | None = ""
    angellist_url: str | None = ""
    linkedin_url: str | None = ""
    twitter_url: str | None = ""
    facebook_url: str | None = ""
    primary_phone: PrimaryPhone | None = PrimaryPhone()
    languages: list[str] | None = []
    alexa_ranking: int | None = 0
    phone: str | None = ""
    linkedin_uid: str | None = ""
    founded_year: int | None = 0
    publicly_traded_symbol: str | None = ""
    publicly_traded_exchange: str | None = ""
    logo_url: str | None = ""
    chrunchbase_url: str | None = ""
    primary_domain: str | None = ""
    sanitized_phone: str | None = ""
    owned_by_organization_id: str | None = ""
    intent_strength: str | None = ""
    show_intent: bool | None = True
    has_intent_signal_account: bool | None = True
    intent_signal_account: str | None = ""


class Contact(BaseModel):
    """A contact in Apollo"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    contact_roles: list[Any] | None = []
    id: str | None = ""
    first_name: str | None = ""
    last_name: str | None = ""
    name: str | None = ""
    linkedin_url: str | None = ""
    title: str | None = ""
    contact_stage_id: str | None = ""
    owner_id: str | None = ""
    creator_id: str | None = ""
    person_id: str | None = ""
    email_needs_tickling: bool | None = True
    organization_name: str | None = ""
    source: str | None = ""
    original_source: str | None = ""
    organization_id: str | None = ""
    headline: str | None = ""
    photo_url: str | None = ""
    present_raw_address: str | None = ""
    linkededin_uid: str | None = ""
    extrapolated_email_confidence: float | None = 0.0
    salesforce_id: str | None = ""
    salesforce_lead_id: str | None = ""
    salesforce_contact_id: str | None = ""
    saleforce_account_id: str | None = ""
    crm_owner_id: str | None = ""
    created_at: str | None = ""
    emailer_campaign_ids: list[str] | None = []
    direct_dial_status: str | None = ""
    direct_dial_enrichment_failed_at: str | None = ""
    email_status: str | None = ""
    email_source: str | None = ""
    account_id: str | None = ""
    last_activity_date: str | None = ""
    hubspot_vid: str | None = ""
    hubspot_company_id: str | None = ""
    crm_id: str | None = ""
    sanitized_phone: str | None = ""
    merged_crm_ids: str | None = ""
    updated_at: str | None = ""
    queued_for_crm_push: bool | None = True
    suggested_from_rule_engine_config_id: str | None = ""
    email_unsubscribed: str | None = ""
    label_ids: list[Any] | None = []
    has_pending_email_arcgate_request: bool | None = True
    has_email_arcgate_request: bool | None = True
    existence_level: str | None = ""
    email: str | None = ""
    email_from_customer: str | None = ""
    typed_custom_fields: list[TypedCustomField] | None = []
    custom_field_errors: Any | None = {}
    salesforce_record_id: str | None = ""
    crm_record_url: str | None = ""
    email_status_unavailable_reason: str | None = ""
    email_true_status: str | None = ""
    updated_email_true_status: bool | None = True
    contact_rule_config_statuses: list[RuleConfigStatus] | None = []
    source_display_name: str | None = ""
    twitter_url: str | None = ""
    contact_campaign_statuses: list[ContactCampaignStatus] | None = []
    state: str | None = ""
    city: str | None = ""
    country: str | None = ""
    account: Account | None = Account()
    contact_emails: list[ContactEmail] | None = []
    organization: Organization | None = Organization()
    employment_history: list[EmploymentHistory] | None = []
    time_zone: str | None = ""
    intent_strength: str | None = ""
    show_intent: bool | None = True
    phone_numbers: list[PhoneNumber] | None = []
    account_phone_note: str | None = ""
    free_domain: bool | None = True
    is_likely_to_engage: bool | None = True
    email_domain_catchall: bool | None = True
    contact_job_change_event: str | None = ""


class SearchOrganizationsRequest(BaseModel):
    """Request for Apollo's search organizations API"""

    organization_num_employees_range: list[int] | None = SchemaField(
        description="""The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.

Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma.""",
        default=[0, 1000000],
    )

    organization_locations: list[str] | None = SchemaField(
        description="""The location of the company headquarters. You can search across cities, US states, and countries.

If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, any Boston-based companies will not appear in your search results, even if they match other parameters.

To exclude companies based on location, use the organization_not_locations parameter.
""",
        default_factory=list,
    )
    organizations_not_locations: list[str] | None = SchemaField(
        description="""Exclude companies from search results based on the location of the company headquarters. You can use cities, US states, and countries as locations to exclude.

This parameter is useful for ensuring you do not prospect in an undesirable territory. For example, if you use ireland as a value, no Ireland-based companies will appear in your search results.
""",
        default_factory=list,
    )
    q_organization_keyword_tags: list[str] | None = SchemaField(
        description="""Filter search results based on keywords associated with companies. For example, you can enter mining as a value to return only companies that have an association with the mining industry.""",
        default_factory=list,
    )
    q_organization_name: str | None = SchemaField(
        description="""Filter search results to include a specific company name.

If the value you enter for this parameter does not match with a company's name, the company will not appear in search results, even if it matches other parameters. Partial matches are accepted. For example, if you filter by the value marketing, a company called NY Marketing Unlimited would still be eligible as a search result, but NY Market Analysis would not be eligible.""",
        default="",
    )
    organization_ids: list[str] | None = SchemaField(
        description="""The Apollo IDs for the companies you want to include in your search results. Each company in the Apollo database is assigned a unique ID.

To find IDs, identify the values for organization_id when you call this endpoint.""",
        default_factory=list,
    )
    max_results: int | None = SchemaField(
        description="""The maximum number of results to return. If you don't specify this parameter, the default is 100.""",
        default=100,
        ge=1,
        le=50000,
        advanced=True,
    )

    page: int = SchemaField(
        description="""The page number of the Apollo data that you want to retrieve.

Use this parameter in combination with the per_page parameter to make search results for navigable and improve the performance of the endpoint.""",
        default=1,
    )
    per_page: int = SchemaField(
        description="""The number of search results that should be returned for each page. Limited the number of results per page improves the endpoint's performance.

Use the page parameter to search the different pages of data.""",
        default=100,
    )


class SearchOrganizationsResponse(BaseModel):
    """Response from Apollo's search organizations API"""

    breadcrumbs: list[Breadcrumb] | None = []
    partial_results_only: bool | None = True
    has_join: bool | None = True
    disable_eu_prospecting: bool | None = True
    partial_results_limit: int | None = 0
    pagination: Pagination = Pagination(
        page=0, per_page=0, total_entries=0, total_pages=0
    )
    # no listed type on the API docs
    accounts: list[Any] = []
    organizations: list[Organization] = []
    models_ids: list[str] = []
    num_fetch_result: str | None = ""
    derived_params: str | None = ""


class SearchPeopleRequest(BaseModel):
    """Request for Apollo's search people API"""

    person_titles: list[str] | None = SchemaField(
        description="""Job titles held by the people you want to find. For a person to be included in search results, they only need to match 1 of the job titles you add. Adding more job titles expands your search results.

Results also include job titles with the same terms, even if they are not exact matches. For example, searching for marketing manager might return people with the job title content marketing manager.

Use this parameter in combination with the person_seniorities[] parameter to find people based on specific job functions and seniority levels.
""",
        default_factory=list,
        placeholder="marketing manager",
    )
    person_locations: list[str] | None = SchemaField(
        description="""The location where people live. You can search across cities, US states, and countries.

To find people based on the headquarters locations of their current employer, use the organization_locations parameter.""",
        default_factory=list,
    )
    person_seniorities: list[SenorityLevels] | None = SchemaField(
        description="""The job seniority that people hold within their current employer. This enables you to find people that currently hold positions at certain reporting levels, such as Director level or senior IC level.

For a person to be included in search results, they only need to match 1 of the seniorities you add. Adding more seniorities expands your search results.

Searches only return results based on their current job title, so searching for Director-level employees only returns people that currently hold a Director-level title. If someone was previously a Director, but is currently a VP, they would not be included in your search results.

Use this parameter in combination with the person_titles[] parameter to find people based on specific job functions and seniority levels.""",
        default_factory=list,
    )
    organization_locations: list[str] | None = SchemaField(
        description="""The location of the company headquarters for a person's current employer. You can search across cities, US states, and countries.

If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, people that work for the Boston-based company will not appear in your results, even if they match other parameters.

To find people based on their personal location, use the person_locations parameter.""",
        default_factory=list,
    )
    q_organization_domains: list[str] | None = SchemaField(
        description="""The domain name for the person's employer. This can be the current employer or a previous employer. Do not include www., the @ symbol, or similar.

You can add multiple domains to search across companies.

  Examples: apollo.io and microsoft.com""",
        default_factory=list,
    )
    contact_email_statuses: list[ContactEmailStatuses] | None = SchemaField(
        description="""The email statuses for the people you want to find. You can add multiple statuses to expand your search.""",
        default_factory=list,
    )
    organization_ids: list[str] | None = SchemaField(
        description="""The Apollo IDs for the companies (employers) you want to include in your search results. Each company in the Apollo database is assigned a unique ID.

To find IDs, call the Organization Search endpoint and identify the values for organization_id.""",
        default_factory=list,
    )
    organization_num_employees_range: list[int] | None = SchemaField(
        description="""The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.

Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma.""",
        default_factory=list,
    )
    q_keywords: str | None = SchemaField(
        description="""A string of words over which we want to filter the results""",
        default="",
    )
    page: int = SchemaField(
        description="""The page number of the Apollo data that you want to retrieve.

Use this parameter in combination with the per_page parameter to make search results for navigable and improve the performance of the endpoint.""",
        default=1,
    )
    per_page: int = SchemaField(
        description="""The number of search results that should be returned for each page. Limited the number of results per page improves the endpoint's performance.

Use the page parameter to search the different pages of data.""",
        default=100,
    )
    max_results: int | None = SchemaField(
        description="""The maximum number of results to return. If you don't specify this parameter, the default is 100.""",
        default=100,
        ge=1,
        le=50000,
        advanced=True,
    )


class SearchPeopleResponse(BaseModel):
    """Response from Apollo's search people API"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    breadcrumbs: list[Breadcrumb] | None = []
    partial_results_only: bool | None = True
    has_join: bool | None = True
    disable_eu_prospecting: bool | None = True
    partial_results_limit: int | None = 0
    pagination: Pagination = Pagination(
        page=0, per_page=0, total_entries=0, total_pages=0
    )
    contacts: list[Contact] = []
    people: list[Contact] = []
    model_ids: list[str] = []
    num_fetch_result: str | None = ""
    derived_params: str | None = ""


class EnrichPersonRequest(BaseModel):
    """Request for Apollo's person enrichment API"""

    person_id: str | None = SchemaField(
        description="Apollo person ID to enrich (most accurate method)",
        default="",
    )
    first_name: str | None = SchemaField(
        description="First name of the person to enrich",
        default="",
    )
    last_name: str | None = SchemaField(
        description="Last name of the person to enrich",
        default="",
    )
    name: str | None = SchemaField(
        description="Full name of the person to enrich",
        default="",
    )
    email: str | None = SchemaField(
        description="Email address of the person to enrich",
        default="",
    )
    domain: str | None = SchemaField(
        description="Company domain of the person to enrich",
        default="",
    )
    company: str | None = SchemaField(
        description="Company name of the person to enrich",
        default="",
    )
    linkedin_url: str | None = SchemaField(
        description="LinkedIn URL of the person to enrich",
        default="",
    )
    organization_id: str | None = SchemaField(
        description="Apollo organization ID of the person's company",
        default="",
    )
    title: str | None = SchemaField(
        description="Job title of the person to enrich",
        default="",
    )
