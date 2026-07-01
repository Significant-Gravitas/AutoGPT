"""Canonical filter enums for DataForB2B search blocks.

Source of truth: the DataForB2B search column spec. Using these as block input
field types makes the columns/operators render as dropdowns in the builder.
"""

from enum import Enum


class FilterOperator(str, Enum):
    EQUALS = "="
    NOT_EQUALS = "!="
    LIKE = "like"
    NOT_LIKE = "not_like"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = ">"
    GREATER_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_OR_EQUAL = "<="
    BETWEEN = "between"


class PeopleColumn(str, Enum):
    # Profile
    first_name = "first_name"
    last_name = "last_name"
    profile_location = "profile_location"
    profile_country = "profile_country"
    profile_industry = "profile_industry"
    follower_count = "follower_count"
    keyword = "keyword"
    # Current job
    current_company = "current_company"
    current_title = "current_title"
    current_job_location = "current_job_location"
    current_company_industry = "current_company_industry"
    current_company_category = "current_company_category"
    current_company_size = "current_company_size"
    current_company_id = "current_company_id"
    current_employment_type = "current_employment_type"
    years_in_current_position = "years_in_current_position"
    years_at_current_company = "years_at_current_company"
    current_company_has_funding = "current_company_has_funding"
    current_company_funding_stage = "current_company_funding_stage"
    current_company_investor = "current_company_investor"
    # Past jobs
    past_company = "past_company"
    past_title = "past_title"
    past_job_location = "past_job_location"
    past_company_industry = "past_company_industry"
    past_company_size = "past_company_size"
    past_company_id = "past_company_id"
    past_employment_type = "past_employment_type"
    years_at_past_company = "years_at_past_company"
    # Skills / education / languages / certifications / experience
    skill = "skill"
    school = "school"
    degree = "degree"
    degree_level = "degree_level"
    field_of_study = "field_of_study"
    language = "language"
    language_iso = "language_iso"
    language_proficiency = "language_proficiency"
    certification = "certification"
    certification_authority = "certification_authority"
    years_of_experience = "years_of_experience"
    num_total_jobs = "num_total_jobs"
    is_currently_employed = "is_currently_employed"


class CompanyColumn(str, Enum):
    # Basic
    company_name = "name"
    tagline = "tagline"
    description = "description"
    domain = "domain"
    universal_name = "universal_name"
    keyword = "keyword"
    industry = "industry"
    # Size
    employee_count = "employee_count"
    # HQ / offices
    country_iso_code = "country_iso_code"
    city = "city"
    region = "region"
    office_country = "office_country"
    office_city = "office_city"
    office_region = "office_region"
    # Growth
    employee_growth_1m = "employee_growth_1m"
    employee_growth_6m = "employee_growth_6m"
    employee_growth_12m = "employee_growth_12m"
    recent_hires_count = "recent_hires_count"
    # Metadata
    founded_year = "founded_year"
    company_type = "company_type"
    follower_count = "follower_count"
    page_verified = "page_verified"
    category = "category"
    # Funding
    last_funding_amount_usd = "last_funding_amount_usd"
    last_funding_date = "last_funding_date"
    funding_stage_normalized = "funding_stage_normalized"
    has_funding = "has_funding"
