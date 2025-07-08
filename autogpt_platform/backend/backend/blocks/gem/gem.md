Gem logo

Introduction
Authentication
Rate Limits
Pagination
Users
Candidates
Candidate Events
Candidate Notes
Candidate Uploaded Resumes
Custom Fields
Custom Field Options
Data Export
Notes
Projects
Project and Candidates Membership
Project Fields
Project Field Options Associations
Project Field Options
Sequences
GET
List sequences
GET
Get sequence by ID
redocly logoAPI docs by Redocly
Gem API (v0)
Download OpenAPI specification:Download

Introduction

Gem API is a RESTful API that provides programmatic methods to access and modify data on the Gem platform. The API is hosted at https://api.gem.com and follows standard REST practices.

Requests are made with standard HTTP methods (GET, POST, PUT, DELETE) to documented URI endpoints with application/json content type.

Responses include standard HTTP response codes (200, 201, 204, 400, 404, etc) and JSON encoded body, where applicable. Errors are also represented with a JSON object in the response body.

NOTE: The Gem API is still in development mode. While stability and compatibility will be carefully maintained as much as possible, changes will likely be introduced without notice. However, breaking changes will always be introduced with a version change and/or prior communication from the support team. Please reach out to your point of contact at Gem to provide feedback.
Authentication

All requests to Gem API is authenticated with your team's API key. Team admins can provision API keys on your team's admin dashboard ("Team Settings"). The API key will be a 40-character alphanumeric string and should be passed under the header X-API-Key. As an example:

curl -X GET \
    -H "X-API-Key: <YOUR_API_KEY>" \
    -H "Content-Type: application/json" \
    https://api.gem.com/v0/users
Rate Limits

By default, each API key is subject to a rate limit of 20 requests per second, with a burst capacity of 500 requests. If you exceed either of these limits, you will receive a 429 Too Many Requests or Limit Exceeded error code.
Pagination

Pagination is supported on GET requests to collection resources by using page and page_size query parameters.

page is 1-indexed. page_size defaults to 20 if a value isn't provided and the maximum allowed value is 100. As an example:

curl -i -X GET \
    -H "X-API-Key: <YOUR_API_KEY>" \
    -H "Content-Type: application/json" \
    https://api.gem.com/v0/users?page=2&page_size=10
Responses from paginated collection resources, even if pagination parameters were not explicitly defined, will include a X-Pagination response header with a JSON object in this shape:

{
    "total": 176,
    "total_pages": 18,
    "first_page": 1,
    "last_page": 18,
    "page": 2,
    "previous_page": 1,
    "next_page": 3
}
Users

Operations on user objects
List users

Returns a list of all users on the team and supports pagination.
QUERY PARAMETERS

email	
string <email>
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/users
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"name": "string",
"email": "user@example.com"
}
]
Candidates

Operations on candidate objects
List candidates

Returns a list of all candidates on the team and supports pagination and search parameters created_after, created_before.
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
created_by	
string <ObjectID>
the id of the user who adds the candidate into Gem
email	
string <email> <= 255 characters
If email is provided, the candidates whose email addresses contain email will be returned.
linked_in_handle	
string [ 1 .. 255 ] characters
updated_after	
integer >= 1
If updated_after is provided, only the candidates updated after the provided timestamp will be returned. Please provide a Unix timestamp in seconds.
candidate_ids	
Array of strings <ObjectID>
If candidate_ids is provided, only the candidates with a provided ID will be returned. Please provide a comma-separated string. e.g.: candidate_ids=123,456,789. A maximum of 20 candidate_ids can be provided.
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/candidates
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"created_at": 1,
"created_by": "string",
"last_updated_at": 1,
"candidate_greenhouse_id": "string",
"first_name": "string",
"last_name": "string",
"nickname": "string",
"weblink": "string",
"emails": [],
"phone_number": "string",
"location": "string",
"linked_in_handle": "string",
"profiles": [],
"company": "string",
"title": "string",
"school": "string",
"education_info": [],
"work_info": [],
"custom_fields": [],
"due_date": {},
"project_ids": [],
"sourced_from": "SeekOut",
"gem_source": "SeekOut"
}
]
Create a new candidate

Returns the newly created candidate object.
REQUEST BODY SCHEMA: application/json
required

created_by
required
string <ObjectID>
first_name	
string or null <= 255 characters
last_name	
string or null <= 255 characters
nickname	
string or null <= 255 characters
emails	
Array of objects (Email) <= 20 items
linked_in_handle	
string or null <= 255 characters
If linked_in_handle is provided, candidate creation will be de-duplicated. If a candidate with the provided linked_in_handle already exists, a 400 error will be returned with errors containing information on the existing candidate in this shape: {"errors": { "duplicate_candidate": { "id": "string", "linked_in_handle": "string" }}}.
title	
string or null <= 255 characters
company	
string or null <= 255 characters
location	
string or null <= 255 characters
school	
string or null <= 255 characters
education_info	
Array of objects or null (EducationInfo)
work_info	
Array of objects or null (WorkInfo)
profile_urls	
Array of strings or null <url>
If profile_urls is provided with an array of urls, social profiles will be generated based on the provided urls and attached to the candidate.
phone_number	
string or null <= 255 characters
project_ids	
Array of strings or null <ObjectID> <= 20 items
If project_ids is provided with an array of project ids, the candidate will be added into the projects once they are created.
custom_fields	
Array of objects (CustomFieldCandidateMembershipUpdate)
Array of objects containing new custom field values. Only custom fields specified in the array are updated. Click the arrow on the left to find more info about the object.
sourced_from	
any or null
Enum: "SeekOut" "hireEZ" "Starcircle" "Censia" "Consider"
autofill	
boolean
Default: false
Requires linked_in_handle to be non-null. Attempts to fill in any missing fields.
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/candidates
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"created_by": "string",
"first_name": "string",
"last_name": "string",
"nickname": "string",
"emails": [
{}
],
"linked_in_handle": "string",
"title": "string",
"company": "string",
"location": "string",
"school": "string",
"education_info": [
{}
],
"work_info": [
{}
],
"profile_urls": [
"string"
],
"phone_number": "string",
"project_ids": [
"string"
],
"custom_fields": [
{}
],
"sourced_from": "SeekOut",
"autofill": false
}
Response samples

201422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"created_by": "string",
"last_updated_at": 1,
"candidate_greenhouse_id": "string",
"first_name": "string",
"last_name": "string",
"nickname": "string",
"weblink": "string",
"emails": [
{}
],
"phone_number": "string",
"location": "string",
"linked_in_handle": "string",
"profiles": [
{}
],
"company": "string",
"title": "string",
"school": "string",
"education_info": [
{}
],
"work_info": [
{}
],
"custom_fields": [
{}
],
"due_date": {
"date": "2019-08-24",
"user_id": "string",
"note": "string"
},
"project_ids": [
"string"
],
"sourced_from": "SeekOut",
"gem_source": "SeekOut"
}
Get candidate by ID

Returns the candidate object with the corresponding ID.
PATH PARAMETERS

candidate_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/candidates/{candidate_id}
Response samples

200default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"created_by": "string",
"last_updated_at": 1,
"candidate_greenhouse_id": "string",
"first_name": "string",
"last_name": "string",
"nickname": "string",
"weblink": "string",
"emails": [
{}
],
"phone_number": "string",
"location": "string",
"linked_in_handle": "string",
"profiles": [
{}
],
"company": "string",
"title": "string",
"school": "string",
"education_info": [
{}
],
"work_info": [
{}
],
"custom_fields": [
{}
],
"due_date": {
"date": "2019-08-24",
"user_id": "string",
"note": "string"
},
"project_ids": [
"string"
],
"sourced_from": "SeekOut",
"gem_source": "SeekOut"
}
Modify a candidate

Modifies a candidate based on ID. Only the fields included in the request argument will be modified.
PATH PARAMETERS

candidate_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

first_name	
string or null <= 255 characters
last_name	
string or null <= 255 characters
nickname	
string or null <= 255 characters
emails	
Array of objects (Email) <= 20 items
title	
string or null <= 255 characters
company	
string or null <= 255 characters
location	
string or null <= 255 characters
school	
string or null <= 255 characters
profile_urls	
Array of strings <url>
If profile_urls is provided with an array of urls, social profiles will be replaced with new profiles based on the provided urls.
phone_number	
string or null <= 255 characters
due_date	
object or null
custom_fields	
Array of objects (CustomFieldCandidateMembershipUpdate)
Array of objects containing custom field values to be updated for the candidate. Only custom fields specified in the array are updated. Click the arrow on the left to find more info about the object.
Responses

200
OK
422
Unprocessable Entity
default
Default error response

PUT
/v0/candidates/{candidate_id}
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"first_name": "string",
"last_name": "string",
"nickname": "string",
"emails": [
{}
],
"title": "string",
"company": "string",
"location": "string",
"school": "string",
"profile_urls": [
"string"
],
"phone_number": "string",
"due_date": {
"date": "2019-08-24",
"user_id": "string",
"note": "string"
},
"custom_fields": [
{}
]
}
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"created_by": "string",
"last_updated_at": 1,
"candidate_greenhouse_id": "string",
"first_name": "string",
"last_name": "string",
"nickname": "string",
"weblink": "string",
"emails": [
{}
],
"phone_number": "string",
"location": "string",
"linked_in_handle": "string",
"profiles": [
{}
],
"company": "string",
"title": "string",
"school": "string",
"education_info": [
{}
],
"work_info": [
{}
],
"custom_fields": [
{}
],
"due_date": {
"date": "2019-08-24",
"user_id": "string",
"note": "string"
},
"project_ids": [
"string"
],
"sourced_from": "SeekOut",
"gem_source": "SeekOut"
}
Delete a candidate

Deletes a candidate based on ID.
PATH PARAMETERS

candidate_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

permanently_remove_contact_info	
boolean
Default: false
Prevent the deleted candidate's contact info from being added back to Gem
on_behalf_of_user_id
required
string <ObjectID>
Responses

204
No Content
422
Unprocessable Entity
default
Default error response

DELETE
/v0/candidates/{candidate_id}
Request samples

Payload
Content type
application/json

Copy
{
"permanently_remove_contact_info": false,
"on_behalf_of_user_id": "string"
}
Response samples

422default
Content type
application/json

Copy
Expand all Collapse all
{
"code": 0,
"status": "string",
"message": "string",
"errors": { }
}
Candidate Events

Operations on candidate events
List candidate events (DEPRECATED). This API is deprecated.

Returns a list of all events for the given candidate and supports pagination and search parameters created_after, created_before.
PATH PARAMETERS

candidate_id
required
string non-empty
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/candidates/{candidate_id}/events
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"candidate_id": "string",
"timestamp": 1,
"user_id": "string",
"on_behalf_of_user_id": "string",
"project_id": "string",
"sequence_id": "string",
"type": "sequences",
"subtype": "first_outreach",
"contact_medium": "inmail",
"reply_status": "interested"
}
]
Create a new candidate event

Returns the newly created event object. NOTE: only manual_touchpoints type event can be created currently.
PATH PARAMETERS

candidate_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

timestamp
required
integer >= 1
user_id
required
string <ObjectID>
on_behalf_of_user_id	
string <ObjectID>
project_id
required
string <ObjectID>
sequence_id
required
string <ObjectID>
type
required
any
Enum: "sequences" "sequence_replies" "manual_touchpoints"
subtype
required
any
Enum: "first_outreach" "follow_up" "reply"
contact_medium
required
any
Enum: "inmail" "phone_call" "text_message" "email" "meeting" "li_connect_request"
reply_status	
any
Enum: "interested" "not_interested" "later"
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/candidates/{candidate_id}/events
Request samples

Payload
Content type
application/json

Copy
{
"timestamp": 1,
"user_id": "string",
"on_behalf_of_user_id": "string",
"project_id": "string",
"sequence_id": "string",
"type": "sequences",
"subtype": "first_outreach",
"contact_medium": "inmail",
"reply_status": "interested"
}
Response samples

201422default
Content type
application/json

Copy
{
"id": "string",
"candidate_id": "string",
"timestamp": 1,
"user_id": "string",
"on_behalf_of_user_id": "string",
"project_id": "string",
"sequence_id": "string",
"type": "sequences",
"subtype": "first_outreach",
"contact_medium": "inmail",
"reply_status": "interested"
}
Get event by event ID

Returns the event object with the corresponding candidate ID and event ID.
PATH PARAMETERS

candidate_id
required
string non-empty
event_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/candidates/{candidate_id}/events/{event_id}
Response samples

200default
Content type
application/json

Copy
{
"id": "string",
"candidate_id": "string",
"timestamp": 1,
"user_id": "string",
"on_behalf_of_user_id": "string",
"project_id": "string",
"sequence_id": "string",
"type": "sequences",
"subtype": "first_outreach",
"contact_medium": "inmail",
"reply_status": "interested"
}
Delete an event by ID

Deletes an event by its ID. NOTE: only manual_touchpoints events can be deleted currently.
PATH PARAMETERS

candidate_id
required
string non-empty
event_id
required
string non-empty
Responses

204
No Content
default
Default error response

DELETE
/v0/candidates/{candidate_id}/events/{event_id}
Response samples

default
Content type
application/json

Copy
Expand all Collapse all
{
"code": 0,
"status": "string",
"message": "string",
"errors": { }
}
List candidate events (DEPRECATED). This API is deprecated.

Returns a list of all events for all candidates and supports pagination and search parameters created_after, created_before.
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/candidates/events
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"candidate_id": "string",
"timestamp": 1,
"user_id": "string",
"on_behalf_of_user_id": "string",
"project_id": "string",
"sequence_id": "string",
"type": "sequences",
"subtype": "first_outreach",
"contact_medium": "inmail",
"reply_status": "interested"
}
]
Candidate Notes

Operations on candidate notes
List notes that belong to a candidate

Returns a list of note objects on the given candidate and supports pagination and search parameters created_after, created_before.
PATH PARAMETERS

candidate_id
required
string non-empty
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/candidates/{candidate_id}/notes
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"candidate_id": "string",
"user_id": "string",
"timestamp": 1,
"is_private": false,
"content": "string"
}
]
Candidate Uploaded Resumes

Operations on resumes uploaded for candidates
List uploaded resumes for a candidate

Returns a list of all resumes uploaded for the candidate given the candidate ID. Resumes can be downloaded through the download_url. Not all resumes have filename stored.
PATH PARAMETERS

candidate_id
required
string non-empty
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/candidates/{candidate_id}/uploaded_resumes
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"candidate_id": "string",
"created_at": 1,
"user_id": "string",
"filename": "string",
"download_url": "string"
}
]
Upload resume for a candidate.

NOTE: Besides candidate_id, you also need to specify user_id in the path to indicate which user is uploading the resume.
PATH PARAMETERS

candidate_id
required
string non-empty
user_id
required
string non-empty
REQUEST BODY SCHEMA: multipart/form-data
required

resume_file
required
string <binary>
Allowed formats are: .pdf, .doc, .docx. File size cannot exceed 10MB.
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/candidates/{candidate_id}/uploaded_resumes/{user_id}
Response samples

201422default
Content type
application/json

Copy
{
"id": "string",
"candidate_id": "string",
"created_at": 1,
"user_id": "string",
"filename": "string",
"download_url": "string"
}
Custom Fields

Operations on custom fields. Custom fields cannot be removed and only can be hidden. Mutations on the custom field values associated with candidates are performed on the Candidates endpoint.
List custom fields

Returns a list of all custom fields. Supports pagination and search parameters created_after, created_before.
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
project_id	
string or null <ObjectID>
Only applicable when scope is project.
scope	
any
Enum: "team" "project"
Custom fields under team scope apply to all candidates. Custom fields under project scope apply to candidates in a specific project.
is_hidden	
boolean
hidden custom fields in the candidate info are not displayed in the Gem product or not returned when making a query on the Candidates endpoint.
name	
string
Any leading and trailing spaces of the name string will be removed.
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/custom_fields
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"created_at": 1,
"name": "string",
"value_type": "date",
"scope": "team",
"project_id": "string",
"is_hidden": true,
"options": []
}
]
Create a new custom field

Returns the newly created custom field object.

De-duplication: If a custom field with the provided name already exists, an error would be returned in the format of:

{
    "code": 400,
    "errors": {
        "json": {
            "custom_field": {
                "id": "example_custom_field_id",
                "is_hidden": false,
                "name": "multi_select",
                "options": [
                    {
                        "id": "example_option_id",
                        "is_hidden": false,
                        "value": "example_value"
                    },
                ],
                "project_id": "UHJvamVjdDox",
                "scope": "project",
                "value_type": "multi_select"
            }
        }
    },
    "message": "A custom field with the name already existed",
    "status": "Bad Request"
}
REQUEST BODY SCHEMA: application/json
required

name
required
string [ 1 .. 50 ] characters
The name of the custom field is unique in its scope. Any leading and trailing spaces of the name string will be removed.
value_type
required
any
Enum: "date" "text" "single_select" "multi_select"
When value_type is text, value is a string. When value_type is date, value is a string of format yyyy-mm-dd. When value_type is single_select, value is a string of an option value. When value_type is multi_select, value is an array of strings of option values.
scope
required
any
Enum: "team" "project"
Custom fields under team scope apply to all candidates. Custom fields under project scope apply to candidates in a specific project.
project_id	
string <ObjectID>
Applicable and required when scope is project. If the custom field has project scope, this is the project ID that its associated with.
option_values	
Array of strings [ 1 .. 50 ] items [ items [ 1 .. 50 ] characters ]
Applicable and required when value_type is one of (single_select, multi_select).
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/custom_fields
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"name": "string",
"value_type": "date",
"scope": "team",
"project_id": "string",
"option_values": [
"string"
]
}
Response samples

201422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"name": "string",
"value_type": "date",
"scope": "team",
"project_id": "string",
"is_hidden": true,
"options": [
{}
]
}
Get custom field by ID

Returns the custom field object with the corresponding ID.
PATH PARAMETERS

custom_field_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/custom_fields/{custom_field_id}
Response samples

200default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"name": "string",
"value_type": "date",
"scope": "team",
"project_id": "string",
"is_hidden": true,
"options": [
{}
]
}
Modify a custom field

Modifies a custom field based on ID. Only the fields included in the request argument will be modified.

De-duplication: If a custom field with the provided name already exists, an error would be returned in the format of:

{
    "code": 400,
    "errors": {
        "json": {
            "custom_field": {
                "id": "example_custom_field_id",
                "is_hidden": false,
                "name": "multi_select",
                "options": [
                    {
                        "id": "example_option_id",
                        "is_hidden": false,
                        "value": "example_value"
                    },
                ],
                "project_id": "UHJvamVjdDox",
                "scope": "project",
                "value_type": "multi_select"
            }
        }
    },
    "message": "A custom field with the name already existed",
    "status": "Bad Request"
}
PATH PARAMETERS

custom_field_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

name	
string [ 1 .. 50 ] characters
The name of the custom field is unique in its scope. Any leading and trailing spaces of the name string will be removed.
is_hidden	
boolean
hidden custom fields in the candidate info are not displayed in the Gem product or not returned when making a query on the Candidates endpoint.
Responses

200
OK
422
Unprocessable Entity
default
Default error response

PATCH
/v0/custom_fields/{custom_field_id}
Request samples

Payload
Content type
application/json

Copy
{
"name": "string",
"is_hidden": true
}
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"name": "string",
"value_type": "date",
"scope": "team",
"project_id": "string",
"is_hidden": true,
"options": [
{}
]
}
Custom Field Options

Operations on options in single_select and multiple_select custom fields. Options cannot be deleted and only can be hidden.
List options in a custom field

Returns a list of all custom fields options in a single_select or multi_select custom field.
PATH PARAMETERS

custom_field_id
required
string non-empty
QUERY PARAMETERS

value	
string
Value of an option in the custom field. Any leading and trailing spaces of the option value will be removed.
is_hidden	
boolean
hidden options are not selectable in the Gem product. Candidate custom field values with hidden options are not returned when making a query to the Candidates endpoint.
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/custom_fields/{custom_field_id}/options
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"value": "string",
"is_hidden": true
}
]
Add options to a custom field

Returns the newly created options in a single_select or multi_select custom field.

De-duplication: If options with the values already existed, a 400 error will be returned in the following format.

{
    "code": 400,
    "errors": {
        "json": {
            "options": [
                {
                    "id": "example_option_id",
                    "is_hidden": false,
                    "value": "example_value"
                }
            ]
        }
    },
    "message": "Custom field options with the values already existed.",
    "status": "Bad Request"
}
PATH PARAMETERS

custom_field_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

option_values
required
Array of strings non-empty [ items [ 1 .. 50 ] characters ]
An array of new option values to be created. Any leading and trailing spaces of the new option values will be removed.
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/custom_fields/{custom_field_id}/options
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"option_values": [
"string"
]
}
Response samples

201422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"value": "string",
"is_hidden": true
}
]
Get an option in a custom field

Returns an option in a single_select or multi_select custom field.
PATH PARAMETERS

custom_field_id
required
string non-empty
option_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/custom_fields/{custom_field_id}/options/{option_id}
Response samples

200default
Content type
application/json

Copy
{
"id": "string",
"value": "string",
"is_hidden": true
}
Modify an option in a custom field

Returns an option in a single_select or multi_select custom field.

Custom Field Option cannot be deleted and can only be hidden.
PATH PARAMETERS

custom_field_id
required
string non-empty
option_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

is_hidden	
boolean
hidden options are not selectable in the Gem product. Candidate custom field values with hidden options are not returned when making a query to the Candidates endpoint.
Responses

200
OK
422
Unprocessable Entity
default
Default error response

PATCH
/v0/custom_fields/{custom_field_id}/options/{option_id}
Request samples

Payload
Content type
application/json

Copy
{
"is_hidden": true
}
Response samples

200422default
Content type
application/json

Copy
{
"id": "string",
"value": "string",
"is_hidden": true
}
Data Export

Operations related to data exports
Get the most recent data export.

Returns the data export information for the most recent export.
Responses

200
OK
default
Default error response

GET
/v0/data_export
Response samples

200default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"files": [
{}
]
}
Notes

Operations on note objects
Create a new note

Returns the newly created note object.
REQUEST BODY SCHEMA: application/json
required

candidate_id
required
string <ObjectID>
user_id
required
string <ObjectID>
is_private	
boolean
Default: false
content
required
string <= 10000 characters
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/notes
Request samples

Payload
Content type
application/json

Copy
{
"candidate_id": "string",
"user_id": "string",
"is_private": false,
"content": "string"
}
Response samples

201422default
Content type
application/json

Copy
{
"id": "string",
"candidate_id": "string",
"user_id": "string",
"timestamp": 1,
"is_private": false,
"content": "string"
}
Get note by note ID

Returns the note object with the note ID.
PATH PARAMETERS

note_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/notes/{note_id}
Response samples

200default
Content type
application/json

Copy
{
"id": "string",
"candidate_id": "string",
"user_id": "string",
"timestamp": 1,
"is_private": false,
"content": "string"
}
Delete a note by ID

Deletes a note by its ID.
PATH PARAMETERS

note_id
required
string non-empty
Responses

204
No Content
default
Default error response

DELETE
/v0/notes/{note_id}
Response samples

default
Content type
application/json

Copy
Expand all Collapse all
{
"code": 0,
"status": "string",
"message": "string",
"errors": { }
}
Projects

Operations on project objects
List projects

Returns a list of all projects on the team and supports pagination and search parameters created_after, created_before.
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
user_id	
string <ObjectID>
List all projects owned by this user.
readable_by	
string <ObjectID>
List all projects that this user has read access to.
writable_by	
string <ObjectID>
List all projects that this user has write access to, that is, all the projects that this user can add/remove candidates
is_archived	
boolean
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/projects
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"created_at": 1,
"user_id": "string",
"name": "string",
"privacy_type": "confidential",
"description": "string",
"is_archived": true,
"project_fields": [],
"context": "string"
}
]
Create a new project

Returns the newly created project object.
REQUEST BODY SCHEMA: application/json
required

user_id
required
string <ObjectID>
The user id of the project owner.
name
required
string <= 255 characters
privacy_type	
any
Default: "personal"
Enum: "confidential" "personal" "shared"
description	
string or null <= 2000 characters
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/projects
Request samples

Payload
Content type
application/json

Copy
{
"user_id": "string",
"name": "string",
"privacy_type": "confidential",
"description": "string"
}
Response samples

201422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"user_id": "string",
"name": "string",
"privacy_type": "confidential",
"description": "string",
"is_archived": true,
"project_fields": [
{}
],
"context": "string"
}
Get project by ID

Returns the project object with the corresponding ID
PATH PARAMETERS

project_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/projects/{project_id}
Response samples

200default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"user_id": "string",
"name": "string",
"privacy_type": "confidential",
"description": "string",
"is_archived": true,
"project_fields": [
{}
],
"context": "string"
}
Modify a project

Modifies a project based on ID. Only the fields included in the request argument will be modified.
PATH PARAMETERS

project_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

user_id	
string <ObjectID>
The user id of the project owner.
name	
string <= 255 characters
privacy_type	
any
Enum: "confidential" "personal" "shared"
description	
string or null <= 2000 characters
is_archived	
boolean
Responses

200
OK
422
Unprocessable Entity
default
Default error response

PATCH
/v0/projects/{project_id}
Request samples

Payload
Content type
application/json

Copy
{
"user_id": "string",
"name": "string",
"privacy_type": "confidential",
"description": "string",
"is_archived": true
}
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"created_at": 1,
"user_id": "string",
"name": "string",
"privacy_type": "confidential",
"description": "string",
"is_archived": true,
"project_fields": [
{}
],
"context": "string"
}
Project and Candidates Membership

Operations on managing candidates in a project
List candidates in a project

Returns a list of all candidates in the project given the project_id. Supports search parameters added_after, added_before.
PATH PARAMETERS

project_id
required
string non-empty
QUERY PARAMETERS

added_after	
integer >= 1
added_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/projects/{project_id}/candidates
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"candidate_id": "string",
"added_at": 1
}
]
Add candidates to a project

Add candidates into a project.

Returns 204 no content if candidates are successfully added to the project.

If candidates are already in the project, a 400 error will be returned with errors containing information on the candidates already in the project in this shape:

{
    "code": 400,
    "errors": {
        "json": {
            "candidate_ids": [
                "string"
            ]
        }
    },
    "message": "Candidates with the ids are already in the project",
    "status": "Bad Request"
}
PATH PARAMETERS

project_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

candidate_ids
required
Array of strings <ObjectID> [ 1 .. 1000 ] items
user_id	
string <ObjectID>
User performing the update. If included, the user must have write access to the project.
Responses

204
No Content
422
Unprocessable Entity
default
Default error response

PUT
/v0/projects/{project_id}/candidates
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"candidate_ids": [
"string"
],
"user_id": "string"
}
Response samples

422default
Content type
application/json

Copy
Expand all Collapse all
{
"code": 0,
"status": "string",
"message": "string",
"errors": { }
}
Remove candidates from a project

Remove candidates from a project.

Returns 204 no content if candidates are successfully removed from the project.

If candidates are not in the project, a 400 error will be returned with errors containing information on the candidates not in the project in this shape:

{
    "code": 400,
    "errors": {
        "json": {
            "candidate_ids": [
                "string"
            ]
        }
    },
    "message": "Candidates with the ids are not in the project",
    "status": "Bad Request"
}
PATH PARAMETERS

project_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

candidate_ids
required
Array of strings <ObjectID> [ 1 .. 1000 ] items
user_id	
string <ObjectID>
User performing the update. If included, the user must have write access to the project.
Responses

204
No Content
422
Unprocessable Entity
default
Default error response

DELETE
/v0/projects/{project_id}/candidates
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"candidate_ids": [
"string"
],
"user_id": "string"
}
Response samples

422default
Content type
application/json

Copy
Expand all Collapse all
{
"code": 0,
"status": "string",
"message": "string",
"errors": { }
}
Fetch membership log with filters

This endpoint allows fetching the membership log for a project or candidate. At least one filter (project_id or candidate_id) must be specified.
QUERY PARAMETERS

changed_after	
integer >= 1
changed_before	
integer >= 1
project_id	
string
candidate_id	
string
sort	
any
Enum: "asc" "desc"
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/project_candidate_membership_log
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"candidate_id": "string",
"project_id": "string",
"action": "string",
"timestamp": 1
}
]
Project Fields

Operations on project fields. Project fields cannot be removed and can only be hidden. Mutations on the field values associated with projects are performed on the Projects endpoint.
List project fields

Returns a list of all project fields. Supports pagination and search parameters created_after, created_before.
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
is_hidden	
boolean
Hidden project fields in projects are not displayed in the Gem product, nor are they returned when making a query on the Projects endpoint.
is_required	
boolean
name	
string
The name of the project field.
field_type	
any
Enum: "text" "single_select" "multi_select"
Specifies the project field type. For text, the value is a string. For single_select, the value is a string representing a selected option. For multi_select, the value is an array of strings, each representing a selected option.
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/project_fields
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"name": "string",
"field_type": "text",
"user_id": "string",
"options": [],
"is_required": true,
"is_hidden": true
}
]
Create project field

Create a new project field.
REQUEST BODY SCHEMA: application/json
required

name
required
string [ 1 .. 255 ] characters
The name of the project field to create. Must differ from existing project field names.
field_type
required
any
Enum: "text" "single_select" "multi_select"
The type of the project field. Options are text, single_select, or multi_select.
options	
Array of strings non-empty [ items [ 1 .. 255 ] characters ]
The options of the project field.Only applicable when value_type is one of (single_select, multi_select).
is_required	
boolean
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/project_fields
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"name": "string",
"field_type": "text",
"options": [
"string"
],
"is_required": true
}
Response samples

201422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"name": "string",
"field_type": "text",
"user_id": "string",
"options": [
{}
],
"is_required": true,
"is_hidden": true
}
Get project field by id

Returns a project field with the coresponding id.
PATH PARAMETERS

project_field_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/project_fields/{project_field_id}
Response samples

200default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"name": "string",
"field_type": "text",
"user_id": "string",
"options": [
{}
],
"is_required": true,
"is_hidden": true
}
Update project field

Update a project field with the corresponding id.
PATH PARAMETERS

project_field_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

name	
string [ 1 .. 255 ] characters
The name of the project field. Must be unique across entire team.
is_required	
boolean
is_hidden	
boolean
When is_hidden is true, field options are not displayed in the Gem product, nor are they returned when making a query on the Projects endpoint.
Responses

200
OK
422
Unprocessable Entity
default
Default error response

PATCH
/v0/project_fields/{project_field_id}
Request samples

Payload
Content type
application/json

Copy
{
"name": "string",
"is_required": true,
"is_hidden": true
}
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
{
"id": "string",
"name": "string",
"field_type": "text",
"user_id": "string",
"options": [
{}
],
"is_required": true,
"is_hidden": true
}
Project Field Options Associations

Manage project field options across all types, including single_select, multiple_select, and text. Supports adding or removing option values to/from project fields, facilitating dynamic updates to project configurations.
List project field options

Returns a list of all options of a project field.
PATH PARAMETERS

project_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/projects/{project_id}/project_field_options
Response samples

200default
Content type
application/json

Copy
Expand all Collapse all
[
{
"project_field_id": "string",
"option_id": "string",
"text": "string",
"field_type": "text",
"is_hidden": true,
"is_required": true
}
]
Add project field option

Add a project field option to a project field.
PATH PARAMETERS

project_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

project_field_id
required
string <ObjectID>
operation
required
string
Enum: "add" "remove"
Specifies the operation on project field options: add or remove. For remove operations on single_select or text fields, options is not required. For multi_select, specify option values to remove.
options	
Array of strings <ObjectID> non-empty
List of option id values. Required for add operations on single_select and multi_select fields. For remove on multi_select, include values to be removed.
text	
string non-empty
The text of the project field. Required for add operations on text fields.
Responses

422
Unprocessable Entity
default
Default error response

POST
/v0/projects/{project_id}/project_field_options
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"project_field_id": "string",
"operation": "add",
"options": [
"string"
],
"text": "string"
}
Response samples

422default
Content type
application/json

Copy
Expand all Collapse all
{
"code": 0,
"status": "string",
"message": "string",
"errors": { }
}
Project Field Options

Operations on options in single_select and multiple_select project fields. Options cannot be deleted and only can be hidden.
List options in a project field

Returns a list of all project fields options in a single_select or multi_select project field.
PATH PARAMETERS

project_field_id
required
string non-empty
QUERY PARAMETERS

value	
string
The value of the project field option.
is_hidden	
boolean
When is_hidden is true, field options are not displayed in the Gem product, nor are they returned when making a query on the Projects endpoint.
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/project_fields/{project_field_id}/options
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"value": "string",
"is_hidden": true
}
]
Create a project field option

Creates a new project field option.
PATH PARAMETERS

project_field_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

options
required
Array of strings non-empty [ items [ 1 .. 255 ] characters ]
Only applicable when value_type is one of (single_select, multi_select). An array of new option values to be created.
Responses

201
Created
422
Unprocessable Entity
default
Default error response

POST
/v0/project_fields/{project_field_id}/options
Request samples

Payload
Content type
application/json

Copy
Expand all Collapse all
{
"options": [
"string"
]
}
Response samples

201422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"value": "string",
"is_hidden": true
}
]
Get a project field option

Returns a project field option by ID.
PATH PARAMETERS

project_field_id
required
string non-empty
project_field_option_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/project_fields/{project_field_id}/options/{project_field_option_id}
Response samples

200default
Content type
application/json

Copy
{
"id": "string",
"value": "string",
"is_hidden": true
}
Update a project field option

Updates a project field option's visibility by ID.
PATH PARAMETERS

project_field_id
required
string non-empty
project_field_option_id
required
string non-empty
REQUEST BODY SCHEMA: application/json
required

is_hidden	
boolean
When is_hidden is true, field options are not displayed in the Gem product, nor are they returned when making a query on the Projects endpoint.
Responses

200
OK
422
Unprocessable Entity
default
Default error response

PATCH
/v0/project_fields/{project_field_id}/options/{project_field_option_id}
Request samples

Payload
Content type
application/json

Copy
{
"is_hidden": true
}
Response samples

200422default
Content type
application/json

Copy
{
"id": "string",
"value": "string",
"is_hidden": true
}
Sequences

Operations on sequence objects
List sequences

Returns a list of sequences that match all of the provided search criteria.
QUERY PARAMETERS

created_after	
integer >= 1
created_before	
integer >= 1
sort	
any
Enum: "asc" "desc"
user_id	
string <ObjectID>
page	
integer >= 1
Default: 1
page_size	
integer [ 1 .. 100 ]
Default: 20
Responses

200
OK
422
Unprocessable Entity
default
Default error response

GET
/v0/sequences
Response samples

200422default
Content type
application/json

Copy
Expand all Collapse all
[
{
"id": "string",
"created_at": 1,
"name": "string",
"user_id": "string"
}
]
Get sequence by ID

Returns the sequence object with the corresponding ID.
PATH PARAMETERS

sequence_id
required
string non-empty
Responses

200
OK
default
Default error response

GET
/v0/sequences/{sequence_id}
Response samples

200default
Content type
application/json

Copy
{
"id": "string",
"created_at": 1,
"name": "string",
"user_id": "string"
}
