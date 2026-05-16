from fastapi import APIRouter, HTTPException
from models.deal import DealState
import state as app_state

router = APIRouter()


@router.post("/")
def create_deal(deal: DealState):
    app_state.deals_store[deal.deal_id] = deal.model_dump()
    return {"deal_id": deal.deal_id}


@router.get("/{deal_id}")
def get_deal(deal_id: str):
    if deal_id not in app_state.deals_store:
        raise HTTPException(status_code=404, detail="Deal not found")
    return app_state.deals_store[deal_id]


@router.put("/{deal_id}")
def update_deal(deal_id: str, deal: DealState):
    deal.deal_id = deal_id
    app_state.deals_store[deal_id] = deal.model_dump()
    return {"deal_id": deal_id}


@router.delete("/{deal_id}")
def delete_deal(deal_id: str):
    app_state.deals_store.pop(deal_id, None)
    return {"deleted": deal_id}


@router.get("/")
def list_deals():
    return [
        {"deal_id": k, "name": v.get("name", ""), "purchase_price": v.get("property_info", {}).get("purchase_price", 0)}
        for k, v in app_state.deals_store.items()
    ]
