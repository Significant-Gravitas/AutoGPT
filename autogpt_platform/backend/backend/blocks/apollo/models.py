from enum import Enum
from typing import Any, Optional

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

    number: Optional[str] = ""
    source: Optional[str] = ""
    sanitized_number: Optional[str] = ""


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

    _id: Optional[str] = ""
    created_at: Optional[str] = ""
    rule_action_config_id: Optional[str] = ""
    rule_config_id: Optional[str] = ""
    status_cd: Optional[str] = ""
    updated_at: Optional[str] = ""
    id: Optional[str] = ""
    key: Optional[str] = ""


class ContactCampaignStatus(BaseModel):
    """A contact campaign status in Apollo"""

    id: Optional[str] = ""
    emailer_campaign_id: Optional[str] = ""
    send_email_from_user_id: Optional[str] = ""
    inactive_reason: Optional[str] = ""
    status: Optional[str] = ""
    added_at: Optional[str] = ""
    added_by_user_id: Optional[str] = ""
    finished_at: Optional[str] = ""
    paused_at: Optional[str] = ""
    auto_unpause_at: Optional[str] = ""
    send_email_from_email_address: Optional[str] = ""
    send_email_from_email_account_id: Optional[str] = ""
    manually_set_unpause: Optional[str] = ""
    failure_reason: Optional[str] = ""
    current_step_id: Optional[str] = ""
    in_response_to_emailer_message_id: Optional[str] = ""
    cc_emails: Optional[str] = ""
    bcc_emails: Optional[str] = ""
    to_emails: Optional[str] = ""


class Account(BaseModel):
    """An account in Apollo"""

    id: Optional[str] = ""
    name: Optional[str] = ""
    website_url: Optional[str] = ""
    blog_url: Optional[str] = ""
    angellist_url: Optional[str] = ""
    linkedin_url: Optional[str] = ""
    twitter_url: Optional[str] = ""
    facebook_url: Optional[str] = ""
    primary_phone: Optional[PrimaryPhone] = PrimaryPhone()
    languages: Optional[list[str]] = []
    alexa_ranking: Optional[int] = 0
    phone: Optional[str] = ""
    linkedin_uid: Optional[str] = ""
    founded_year: Optional[int] = 0
    publicly_traded_symbol: Optional[str] = ""
    publicly_traded_exchange: Optional[str] = ""
    logo_url: Optional[str] = ""
    chrunchbase_url: Optional[str] = ""
    primary_domain: Optional[str] = ""
    domain: Optional[str] = ""
    team_id: Optional[str] = ""
    organization_id: Optional[str] = ""
    account_stage_id: Optional[str] = ""
    source: Optional[str] = ""
    original_source: Optional[str] = ""
    creator_id: Optional[str] = ""
    owner_id: Optional[str] = ""
    created_at: Optional[str] = ""
    phone_status: Optional[str] = ""
    hubspot_id: Optional[str] = ""
    salesforce_id: Optional[str] = ""
    crm_owner_id: Optional[str] = ""
    parent_account_id: Optional[str] = ""
    sanitized_phone: Optional[str] = ""
    # no listed type on the API docs
    account_playbook_statues: Optional[list[Any]] = []
    account_rule_config_statuses: Optional[list[RuleConfigStatus]] = []
    existence_level: Optional[str] = ""
    label_ids: Optional[list[str]] = []
    typed_custom_fields: Optional[Any] = {}
    custom_field_errors: Optional[Any] = {}
    modality: Optional[str] = ""
    source_display_name: Optional[str] = ""
    salesforce_record_id: Optional[str] = ""
    crm_record_url: Optional[str] = ""


class ContactEmail(BaseModel):
    """A contact email in Apollo"""

    email: Optional[str] = ""
    email_md5: Optional[str] = ""
    email_sha256: Optional[str] = ""
    email_status: Optional[str] = ""
    email_source: Optional[str] = ""
    extrapolated_email_confidence: Optional[str] = ""
    position: Optional[int] = 0
    email_from_customer: Optional[str] = ""
    free_domain: Optional[bool] = True


class EmploymentHistory(BaseModel):
    """An employment history in Apollo"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    _id: Optional[str] = ""
    created_at: Optional[str] = ""
    current: Optional[bool] = False
    degree: Optional[str] = ""
    description: Optional[str] = ""
    emails: Optional[str] = ""
    end_date: Optional[str] = ""
    grade_level: Optional[str] = ""
    kind: Optional[str] = ""
    major: Optional[str] = ""
    organization_id: Optional[str] = ""
    organization_name: Optional[str] = ""
    raw_address: Optional[str] = ""
    start_date: Optional[str] = ""
    title: Optional[str] = ""
    updated_at: Optional[str] = ""
    id: Optional[str] = ""
    key: Optional[str] = ""


class Breadcrumb(BaseModel):
    """A breadcrumb in Apollo"""

    label: Optional[str] = ""
    signal_field_name: Optional[str] = ""
    value: str | list | None = ""
    display_name: Optional[str] = ""


class TypedCustomField(BaseModel):
    """A typed custom field in Apollo"""

    id: Optional[str] = ""
    value: Optional[str] = ""


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

    country_name: Optional[str] = ""
    country_enabled: Optional[bool] = True
    high_risk_calling_enabled: Optional[bool] = True
    potential_high_risk_number: Optional[bool] = True


class PhoneNumber(BaseModel):
    """A phone number in Apollo"""

    raw_number: Optional[str] = ""
    sanitized_number: Optional[str] = ""
    type: Optional[str] = ""
    position: Optional[int] = 0
    status: Optional[str] = ""
    dnc_status: Optional[str] = ""
    dnc_other_info: Optional[str] = ""
    dailer_flags: Optional[DialerFlags] = DialerFlags(
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

    id: Optional[str] = ""
    name: Optional[str] = ""
    website_url: Optional[str] = ""
    blog_url: Optional[str] = ""
    angellist_url: Optional[str] = ""
    linkedin_url: Optional[str] = ""
    twitter_url: Optional[str] = ""
    facebook_url: Optional[str] = ""
    primary_phone: Optional[PrimaryPhone] = PrimaryPhone()
    languages: Optional[list[str]] = []
    alexa_ranking: Optional[int] = 0
    phone: Optional[str] = ""
    linkedin_uid: Optional[str] = ""
    founded_year: Optional[int] = 0
    publicly_traded_symbol: Optional[str] = ""
    publicly_traded_exchange: Optional[str] = ""
    logo_url: Optional[str] = ""
    chrunchbase_url: Optional[str] = ""
    primary_domain: Optional[str] = ""
    sanitized_phone: Optional[str] = ""
    owned_by_organization_id: Optional[str] = ""
    intent_strength: Optional[str] = ""
    show_intent: Optional[bool] = True
    has_intent_signal_account: Optional[bool] = True
    intent_signal_account: Optional[str] = ""


class Contact(BaseModel):
    """A contact in Apollo"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
    )

    contact_roles: Optional[list[Any]] = []
    id: Optional[str] = ""
    first_name: Optional[str] = ""
    last_name: Optional[str] = ""
    name: Optional[str] = ""
    linkedin_url: Optional[str] = ""
    title: Optional[str] = ""
    contact_stage_id: Optional[str] = ""
    owner_id: Optional[str] = ""
    creator_id: Optional[str] = ""
    person_id: Optional[str] = ""
    email_needs_tickling: Optional[bool] = True
    organization_name: Optional[str] = ""
    source: Optional[str] = ""
    original_source: Optional[str] = ""
    organization_id: Optional[str] = ""
    headline: Optional[str] = ""
    photo_url: Optional[str] = ""
    present_raw_address: Optional[str] = ""
    linkededin_uid: Optional[str] = ""
    extrapolated_email_confidence: Optional[float] = 0.0
    salesforce_id: Optional[str] = ""
    salesforce_lead_id: Optional[str] = ""
    salesforce_contact_id: Optional[str] = ""
    saleforce_account_id: Optional[str] = ""
    crm_owner_id: Optional[str] = ""
    created_at: Optional[str] = ""
    emailer_campaign_ids: Optional[list[str]] = []
    direct_dial_status: Optional[str] = ""
    direct_dial_enrichment_failed_at: Optional[str] = ""
    email_status: Optional[str] = ""
    email_source: Optional[str] = ""
    account_id: Optional[str] = ""
    last_activity_date: Optional[str] = ""
    hubspot_vid: Optional[str] = ""
    hubspot_company_id: Optional[str] = ""
    crm_id: Optional[str] = ""
    sanitized_phone: Optional[str] = ""
    merged_crm_ids: Optional[str] = ""
    updated_at: Optional[str] = ""
    queued_for_crm_push: Optional[bool] = True
    suggested_from_rule_engine_config_id: Optional[str] = ""
    email_unsubscribed: Optional[str] = ""
    label_ids: Optional[list[Any]] = []
    has_pending_email_arcgate_request: Optional[bool] = True
    has_email_arcgate_request: Optional[bool] = True
    existence_level: Optional[str] = ""
    email: Optional[str] = ""
    email_from_customer: Optional[str] = ""
    typed_custom_fields: Optional[list[TypedCustomField]] = []
    custom_field_errors: Optional[Any] = {}
    salesforce_record_id: Optional[str] = ""
    crm_record_url: Optional[str] = ""
    email_status_unavailable_reason: Optional[str] = ""
    email_true_status: Optional[str] = ""
    updated_email_true_status: Optional[bool] = True
    contact_rule_config_statuses: Optional[list[RuleConfigStatus]] = []
    source_display_name: Optional[str] = ""
    twitter_url: Optional[str] = ""
    contact_campaign_statuses: Optional[list[ContactCampaignStatus]] = []
    state: Optional[str] = ""
    city: Optional[str] = ""
    country: Optional[str] = ""
    account: Optional[Account] = Account()
    contact_emails: Optional[list[ContactEmail]] = []
    organization: Optional[Organization] = Organization()
    employment_history: Optional[list[EmploymentHistory]] = []
    time_zone: Optional[str] = ""
    intent_strength: Optional[str] = ""
    show_intent: Optional[bool] = True
    phone_numbers: Optional[list[PhoneNumber]] = []
    account_phone_note: Optional[str] = ""
    free_domain: Optional[bool] = True
    is_likely_to_engage: Optional[bool] = True
    email_domain_catchall: Optional[bool] = True
    contact_job_change_event: Optional[str] = ""


class SearchOrganizationsRequest(BaseModel):
    """Request for Apollo's search organizations API"""

    organization_num_employees_range: Optional[list[int]] = SchemaField(
        description="""The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.

Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma.""",
        default=[0, 1000000],
    )

    organization_locations: Optional[list[str]] = SchemaField(
        description="""The location of the company headquarters. You can search across cities, US states, and countries.

If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, any Boston-based companies will not appear in your search results, even if they match other parameters.

To exclude companies based on location, use the organization_not_locations parameter.
""",
        default_factory=list,
    )
    organizations_not_locations: Optional[list[str]] = SchemaField(
        description="""Exclude companies from search results based on the location of the company headquarters. You can use cities, US states, and countries as locations to exclude.

This parameter is useful for ensuring you do not prospect in an undesirable territory. For example, if you use ireland as a value, no Ireland-based companies will appear in your search results.
""",
        default_factory=list,
    )
    q_organization_keyword_tags: Optional[list[str]] = SchemaField(
        description="""Filter search results based on keywords associated with companies. For example, you can enter mining as a value to return only companies that have an association with the mining industry.""",
        default_factory=list,
    )
    q_organization_name: Optional[str] = SchemaField(
        description="""Filter search results to include a specific company name.

If the value you enter for this parameter does not match with a company's name, the company will not appear in search results, even if it matches other parameters. Partial matches are accepted. For example, if you filter by the value marketing, a company called NY Marketing Unlimited would still be eligible as a search result, but NY Market Analysis would not be eligible.""",
        default="",
    )
    organization_ids: Optional[list[str]] = SchemaField(
        description="""The Apollo IDs for the companies you want to include in your search results. Each company in the Apollo database is assigned a unique ID.

To find IDs, identify the values for organization_id when you call this endpoint.""",
        default_factory=list,
    )
    max_results: Optional[int] = SchemaField(
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

    breadcrumbs: Optional[list[Breadcrumb]] = []
    partial_results_only: Optional[bool] = True
    has_join: Optional[bool] = True
    disable_eu_prospecting: Optional[bool] = True
    partial_results_limit: Optional[int] = 0
    pagination: Pagination = Pagination(
        page=0, per_page=0, total_entries=0, total_pages=0
    )
    # no listed type on the API docs
    accounts: list[Any] = []
    organizations: list[Organization] = []
    models_ids: list[str] = []
    num_fetch_result: Optional[str] = ""
    derived_params: Optional[str] = ""


class SearchPeopleRequest(BaseModel):
    """Request for Apollo's search people API"""

    person_titles: Optional[list[str]] = SchemaField(
        description="""Job titles held by the people you want to find. For a person to be included in search results, they only need to match 1 of the job titles you add. Adding more job titles expands your search results.

Results also include job titles with the same terms, even if they are not exact matches. For example, searching for marketing manager might return people with the job title content marketing manager.

Use this parameter in combination with the person_seniorities[] parameter to find people based on specific job functions and seniority levels.
""",
        default_factory=list,
        placeholder="marketing manager",
    )
    person_locations: Optional[list[str]] = SchemaField(
        description="""The location where people live. You can search across cities, US states, and countries.

To find people based on the headquarters locations of their current employer, use the organization_locations parameter.""",
        default_factory=list,
    )
    person_seniorities: Optional[list[SenorityLevels]] = SchemaField(
        description="""The job seniority that people hold within their current employer. This enables you to find people that currently hold positions at certain reporting levels, such as Director level or senior IC level.

For a person to be included in search results, they only need to match 1 of the seniorities you add. Adding more seniorities expands your search results.

Searches only return results based on their current job title, so searching for Director-level employees only returns people that currently hold a Director-level title. If someone was previously a Director, but is currently a VP, they would not be included in your search results.

Use this parameter in combination with the person_titles[] parameter to find people based on specific job functions and seniority levels.""",
        default_factory=list,
    )
    organization_locations: Optional[list[str]] = SchemaField(
        description="""The location of the company headquarters for a person's current employer. You can search across cities, US states, and countries.

If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, people that work for the Boston-based company will not appear in your results, even if they match other parameters.

To find people based on their personal location, use the person_locations parameter.""",
        default_factory=list,
    )
    q_organization_domains: Optional[list[str]] = SchemaField(
        description="""The domain name for the person's employer. This can be the current employer or a previous employer. Do not include www., the @ symbol, or similar.

You can add multiple domains to search across companies.

  Examples: apollo.io and microsoft.com""",
        default_factory=list,
    )
    contact_email_statuses: Optional[list[ContactEmailStatuses]] = SchemaField(
        description="""The email statuses for the people you want to find. You can add multiple statuses to expand your search.""",
        default_factory=list,
    )
    organization_ids: Optional[list[str]] = SchemaField(
        description="""The Apollo IDs for the companies (employers) you want to include in your search results. Each company in the Apollo database is assigned a unique ID.

To find IDs, call the Organization Search endpoint and identify the values for organization_id.""",
        default_factory=list,
    )
    organization_num_employees_range: Optional[list[int]] = SchemaField(
        description="""The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.

Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma.""",
        default_factory=list,
    )
    q_keywords: Optional[str] = SchemaField(
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
    max_results: Optional[int] = SchemaField(
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

    breadcrumbs: Optional[list[Breadcrumb]] = []
    partial_results_only: Optional[bool] = True
    has_join: Optional[bool] = True
    disable_eu_prospecting: Optional[bool] = True
    partial_results_limit: Optional[int] = 0
    pagination: Pagination = Pagination(
        page=0, per_page=0, total_entries=0, total_pages=0
    )
    contacts: list[Contact] = []
    people: list[Contact] = []
    model_ids: list[str] = []
    num_fetch_result: Optional[str] = ""
    derived_params: Optional[str] = ""


class EnrichPersonRequest(BaseModel):
    """Request for Apollo's person enrichment API"""

    person_id: Optional[str] = SchemaField(
        description="Apollo person ID to enrich (most accurate method)",
        default="",
    )
    first_name: Optional[str] = SchemaField(
        description="First name of the person to enrich",
        default="",
    )
    last_name: Optional[str] = SchemaField(
        description="Last name of the person to enrich",
        default="",
    )
    name: Optional[str] = SchemaField(
        description="Full name of the person to enrich",
        default="",
    )
    email: Optional[str] = SchemaField(
        description="Email address of the person to enrich",
        default="",
    )
    domain: Optional[str] = SchemaField(
        description="Company domain of the person to enrich",
        default="",
    )
    company: Optional[str] = SchemaField(
        description="Company name of the person to enrich",
        default="",
    )
    linkedin_url: Optional[str] = SchemaField(
        description="LinkedIn URL of the person to enrich",
        default="",
    )
    organization_id: Optional[str] = SchemaField(
        description="Apollo organization ID of the person's company",
        default="",
    )
    title: Optional[str] = SchemaField(
        description="Job title of the person to enrich",
        default="",
    )
