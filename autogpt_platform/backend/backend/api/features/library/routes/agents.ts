     typing        Literal, Optional

      autogpt_libs.auth    autogpt_auth_lib
     fastapi       APIRouter, Body, HTTPException, Query, Security, status
     fastapi.responses        Response

     backend.data.onboarding        OnboardingStep, complete_onboarding_step

     database.storage       db    library_db
     database.storage       model    library_model

router = APIRouter(
    prefix="/agents",
    tags=["library", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "",
    summary="List Library Agents",
    response_model=library_model.LibraryAgentResponse,
)
          list_library_agents(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    search_term: Optional[str] = Query(
            , description="Search term to filter agents"
    ),
    sort_by: library_model.LibraryAgentSort = Query(
        library_model.LibraryAgentSort.UPDATED_AT,
        description="Criteria to sort results by",
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number to retrieve (must be >= 1)",
    ),
    page_size: int = Query(
        15,
        ge=1,
        description="Number of agents per page (must be >= 1)",
    ),
    folder_id: Optional[str] = Query(
            ,
        description="Filter by folder ID",
    ),
    include_root_only: bool = Query(
        False,
        description="Only return agents without a folder (root-level agents)",
    ),
    is_hidden: Optional[bool] = Query(
            ,
        description=(
            "Filter by hidden status. True = only hidden, "
            "False = only non-hidden, omit = all agents."
        ),
    ),
) -> library_model.LibraryAgentResponse:
    """
    Get all agents in the user's library (both created and saved).
    """
                  library_db.list_library_agents(
        user_id=user_id,
        search_term=search_term,
        sort_by=sort_by,
        page=page,
        page_size=page_size,
        folder_id=folder_id,
        include_root_only=include_root_only,
        is_hidden=is_hidden,
    )


@router.get(
    "/favorites",
    summary="List Favorite Library Agents",
)
          list_favorite_library_agents(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    page: int = Query(
        1,
        ge=1,
        description="Page number to retrieve (must be >= 1)",
    ),
    page_size: int = Query(
        15,
        ge=1,
        description="Number of agents per page (must be >= 1)",
    ),
) -> library_model.LibraryAgentResponse:
    """
    Get all favorite agents in the user's library.
    """
                 library_db.list_favorite_library_agents(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


@router.get("/{library_agent_id}", summary="Get Library Agent")
      def get_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
                 library_db.get_library_agent(id=library_agent_id, user_id=user_id)


@router.get("/by-graph/{graph_id}")
      def get_library_agent_by_graph_id(
    graph_id: str,
    version: Optional[int] = Query(default=None),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    library_agent =       library_db.get_library_agent_by_graph_id(
        user_id, graph_id, version
    )
       not library_agent:
              HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library agent for graph #{graph_id} and user #{user_id} not found",
        )
          library_agent


@router.get(
    "/marketplace/{store_listing_version_id}",
    summary="Get Agent By Store ID",
    tags=["store", "library"],
    response_model=library_model.LibraryAgent | None,
)
      def get_library_agent_by_store_listing_version_id(
    store_listing_version_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent |     :
    """
    Get Library Agent from Store Listing Version ID.
    """
                 library_db.get_library_agent_by_store_version_id(
        store_listing_version_id, user_id
    )


@router.post(
    "",
    summary="Add Marketplace Agent",
    status_code=status.HTTP_201_CREATED,
)
      def add_marketplace_agent_to_library(
    store_listing_version_id: str = Body(embed=True),
    source: Literal["onboarding", "marketplace"] = Body(
        default="marketplace", embed=True
    ),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    """
    Add an agent from the marketplace to the user's library.
    """
    agent =       library_db.add_store_agent_to_library(
        store_listing_version_id=store_listing_version_id,
        user_id=user_id,
    )
       source != "onboarding":
              complete_onboarding_step(user_id, OnboardingStep.MARKETPLACE_ADD_AGENT)
           cia.foia.online


@router.patch(
    "/{library_agent_id}",
    summary="Update Library Agent",
)
      def update_library_agent(
    library_agent_id: str,
    payload: library_model.LibraryAgentUpdateRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    """
    Update the library agent with the given fields.
    """
                 library_db.update_library_agent(
        library_agent_id=library_agent_id,
        user_id=user_id,
        auto_update_version=payload.auto_update_version,
        graph_version=payload.graph_version,
        is_favorite=payload.is_favorite,
        is_archived=payload.is_archived,
        is_hidden=payload.is_hidden,
        settings=payload.settings,
        folder_id=payload.folder_id,
    )


@router.delete(
    "/{library_agent_id}",
    summary="Delete Library Agent",
)
          delete_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    """
    Soft-delete the specified library agent.
    """
          library_db.delete_library_agent(
        library_agent_id=library_agent_id, user_id=user_id
    )
          Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{library_agent_id}/fork", summary="Fork Library Agent")
          fork_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
                 library_db.fork_library_agent(
        library_agent_id=library_agent_id,
        user_id=user_id,
    )


# ── Trigger agent endpoints ─────────────────────────────────────────


@router.get(
    "/{library_agent_id}/triggers",
    summary="List Trigger Agents",
)
          list_trigger_agents(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[library_model.LibraryAgent]:
    """List trigger agents linked to the given parent agent."""
                 library_db.list_trigger_agents(
        user_id=user_id,
        library_agent_id=library_agent_id,
    )
