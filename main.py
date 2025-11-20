import os
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId

# Database helpers (MongoDB)
from database import db

# -----------------------------
# Utilities
# -----------------------------

def oid(obj: Optional[str]) -> Optional[ObjectId]:
    if obj is None:
        return None
    try:
        return ObjectId(obj)
    except Exception:
        return None


def to_str_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = {**doc}
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # normalize ObjectId refs to string
    for k, v in list(d.items()):
        if isinstance(v, ObjectId):
            d[k] = str(v)
    return d


# -----------------------------
# Pydantic Schemas (API layer)
# -----------------------------
class CategoryIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class CategoryOut(BaseModel):
    id: str
    name: str


class ProductIn(BaseModel):
    name: str
    sku: Optional[str] = None
    barcode: Optional[str] = None
    price: float = 0.0
    cost: float = 0.0
    stock: float = 0.0
    tax_rate: float = 0.0
    active: bool = True
    category_id: Optional[str] = None


class ProductOut(BaseModel):
    id: str
    name: str
    sku: Optional[str]
    barcode: Optional[str]
    price: float
    cost: float
    stock: float
    tax_rate: float
    active: bool
    category_id: Optional[str] = None
    category_name: Optional[str] = None


class CustomerIn(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None


class CustomerOut(BaseModel):
    id: str
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None


class POSItem(BaseModel):
    product_id: Optional[str] = None
    name: str
    qty: float
    price: float
    discount: float = 0.0
    tax_rate: float = 0.0


class PricingRequest(BaseModel):
    items: List[POSItem]
    order_discount: float = 0.0


class PricingResponse(BaseModel):
    subtotal: float
    discount: float
    tax: float
    total: float


class CheckoutRequest(BaseModel):
    items: List[POSItem]
    order_discount: float = 0.0
    payment_method: str = "cash"
    customer_id: Optional[str] = None


class OrderItemOut(BaseModel):
    id: str
    product_id: Optional[str] = None
    product_name: str
    qty: float
    price: float
    discount: float
    tax: float
    line_total: float


class OrderOut(BaseModel):
    id: str
    order_number: str
    created_at: datetime
    subtotal: float
    discount: float
    tax: float
    total: float
    payment_method: str
    customer_id: Optional[str] = None
    items: List[OrderItemOut]


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="POS API (Olsera-style) - MongoDB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "POS Backend Running", "driver": "mongodb", "db": os.getenv("DATABASE_NAME")}


@app.get("/health")
def health():
    try:
        db.command("ping")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Helpers
# -----------------------------

def calculate_pricing(items: List[POSItem], order_discount: float) -> PricingResponse:
    subtotal = 0.0
    discount_total = 0.0
    tax_total = 0.0

    for it in items:
        line_subtotal = it.qty * it.price
        line_discount = min(it.discount, line_subtotal)
        taxable_base = max(line_subtotal - line_discount, 0.0)
        line_tax = taxable_base * (it.tax_rate / 100.0)

        subtotal += line_subtotal
        discount_total += line_discount
        tax_total += line_tax

    discount_total += min(order_discount, max(subtotal - discount_total, 0.0))
    total = max(subtotal - discount_total, 0.0) + tax_total

    return PricingResponse(
        subtotal=round(subtotal, 2),
        discount=round(discount_total, 2),
        tax=round(tax_total, 2),
        total=round(total, 2),
    )


def generate_order_number() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    like = f"{today}-"
    last = db["order"].find({"order_number": {"$regex": f"^{like}"}}).sort("order_number", -1).limit(1)
    last_doc = next(last, None)
    if not last_doc:
        return f"{today}-0001"
    try:
        seq = int(str(last_doc.get("order_number")).split("-")[1]) + 1
    except Exception:
        seq = 1
    return f"{today}-{seq:04d}"


# -----------------------------
# Categories
# -----------------------------
@app.get("/categories", response_model=List[CategoryOut])
def list_categories():
    docs = list(db["category"].find({}).sort("name", 1))
    return [CategoryOut(**to_str_id(d)) for d in docs]


@app.post("/categories", response_model=CategoryOut)
def create_category(payload: CategoryIn):
    exists = db["category"].find_one({"name": payload.name})
    if exists:
        raise HTTPException(status_code=400, detail="Category already exists")
    res = db["category"].insert_one({"name": payload.name})
    doc = db["category"].find_one({"_id": res.inserted_id})
    return CategoryOut(**to_str_id(doc))


@app.put("/categories/{category_id}", response_model=CategoryOut)
def update_category(category_id: str, payload: CategoryIn):
    _id = oid(category_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Category not found")
    upd = db["category"].find_one_and_update({"_id": _id}, {"$set": {"name": payload.name}}, return_document=True)
    if not upd:
        raise HTTPException(status_code=404, detail="Category not found")
    return CategoryOut(**to_str_id(upd))


@app.delete("/categories/{category_id}")
def delete_category(category_id: str):
    _id = oid(category_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Category not found")
    res = db["category"].delete_one({"_id": _id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "deleted"}


# -----------------------------
# Products
# -----------------------------
@app.get("/products", response_model=List[ProductOut])
def list_products(
    q: Optional[str] = Query(None, description="Search by name, sku, barcode"),
    category_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    filt: Dict[str, Any] = {}
    if q:
        filt["$or"] = [
            {"name": {"$regex": q, "$options": "i"}},
            {"sku": {"$regex": q, "$options": "i"}},
            {"barcode": {"$regex": q, "$options": "i"}},
        ]
    if category_id:
        filt["category_id"] = category_id

    cursor = db["product"].find(filt).sort("name", 1).skip(offset).limit(limit)
    docs = list(cursor)

    cat_map: Dict[str, str] = {}
    cat_ids = list({d.get("category_id") for d in docs if d.get("category_id")})
    if cat_ids:
        for c in db["category"].find({"_id": {"$in": [oid(x) for x in cat_ids if oid(x)]}}):
            cat_map[str(c["_id"])] = c.get("name")

    out: List[ProductOut] = []
    for d in docs:
        td = to_str_id(d)
        td["category_name"] = cat_map.get(td.get("category_id"))
        out.append(ProductOut(**td))
    return out


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str):
    _id = oid(product_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Product not found")
    d = db["product"].find_one({"_id": _id})
    if not d:
        raise HTTPException(status_code=404, detail="Product not found")
    td = to_str_id(d)
    if td.get("category_id"):
        cat = db["category"].find_one({"_id": oid(td.get("category_id"))})
        td["category_name"] = cat.get("name") if cat else None
    return ProductOut(**td)


@app.post("/products", response_model=ProductOut)
def create_product(payload: ProductIn):
    if payload.sku and db["product"].find_one({"sku": payload.sku}):
        raise HTTPException(status_code=400, detail="SKU already exists")
    if payload.barcode and db["product"].find_one({"barcode": payload.barcode}):
        raise HTTPException(status_code=400, detail="Barcode already exists")

    doc = payload.model_dump()
    if doc.get("category_id") and not oid(doc["category_id"]):
        doc["category_id"] = None
    res = db["product"].insert_one(doc)
    return get_product(str(res.inserted_id))


@app.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: str, payload: ProductIn):
    _id = oid(product_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Product not found")
    cur = db["product"].find_one({"_id": _id})
    if not cur:
        raise HTTPException(status_code=404, detail="Product not found")

    if payload.sku and payload.sku != cur.get("sku") and db["product"].find_one({"sku": payload.sku}):
        raise HTTPException(status_code=400, detail="SKU already exists")
    if payload.barcode and payload.barcode != cur.get("barcode") and db["product"].find_one({"barcode": payload.barcode}):
        raise HTTPException(status_code=400, detail="Barcode already exists")

    doc = payload.model_dump()
    if doc.get("category_id") and not oid(doc["category_id"]):
        doc["category_id"] = None

    db["product"].update_one({"_id": _id}, {"$set": doc})
    return get_product(product_id)


@app.delete("/products/{product_id}")
def delete_product(product_id: str):
    _id = oid(product_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Product not found")
    res = db["product"].delete_one({"_id": _id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"message": "deleted"}


# -----------------------------
# Customers
# -----------------------------
@app.get("/customers", response_model=List[CustomerOut])
def list_customers():
    docs = list(db["customer"].find({}).sort("name", 1))
    return [CustomerOut(**to_str_id(d)) for d in docs]


@app.post("/customers", response_model=CustomerOut)
def create_customer(payload: CustomerIn):
    res = db["customer"].insert_one(payload.model_dump())
    doc = db["customer"].find_one({"_id": res.inserted_id})
    return CustomerOut(**to_str_id(doc))


@app.put("/customers/{customer_id}", response_model=CustomerOut)
def update_customer(customer_id: str, payload: CustomerIn):
    _id = oid(customer_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Customer not found")
    upd = db["customer"].find_one_and_update({"_id": _id}, {"$set": payload.model_dump()}, return_document=True)
    if not upd:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CustomerOut(**to_str_id(upd))


@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: str):
    _id = oid(customer_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Customer not found")
    res = db["customer"].delete_one({"_id": _id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"message": "deleted"}


# -----------------------------
# POS Pricing & Checkout
# -----------------------------
@app.post("/pos/pricing", response_model=PricingResponse)
def pos_pricing(payload: PricingRequest):
    return calculate_pricing(payload.items, payload.order_discount)


@app.post("/pos/checkout", response_model=OrderOut)
def pos_checkout(payload: CheckoutRequest):
    # Validate products & stocks
    for it in payload.items:
        if it.product_id:
            p = db["product"].find_one({"_id": oid(it.product_id)})
            if not p:
                raise HTTPException(status_code=400, detail=f"Product {it.product_id} not found")
            if p.get("stock") is not None and float(p.get("stock", 0)) < it.qty:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for {p.get('name')}")

    pricing = calculate_pricing(payload.items, payload.order_discount)

    order_doc = {
        "order_number": generate_order_number(),
        "created_at": datetime.now(timezone.utc),
        "subtotal": pricing.subtotal,
        "discount": pricing.discount,
        "tax": pricing.tax,
        "total": pricing.total,
        "payment_method": payload.payment_method,
        "customer_id": payload.customer_id,
        "items": [],
    }

    # Create items and adjust stock
    for it in payload.items:
        line_subtotal = it.qty * it.price
        line_discount = min(it.discount, line_subtotal)
        taxable_base = max(line_subtotal - line_discount, 0.0)
        line_tax = taxable_base * (it.tax_rate / 100.0)
        line_total = taxable_base + line_tax

        oi = {
            "product_id": it.product_id,
            "product_name": it.name,
            "qty": it.qty,
            "price": it.price,
            "discount": round(line_discount, 2),
            "tax": round(line_tax, 2),
            "line_total": round(line_total, 2),
        }
        order_doc["items"].append(oi)

        if it.product_id:
            db["product"].update_one({"_id": oid(it.product_id)}, {"$inc": {"stock": -it.qty}})

    res = db["order"].insert_one(order_doc)
    saved = db["order"].find_one({"_id": res.inserted_id})

    items_out = []
    for i in saved.get("items", []):
        i_out = {**i, "id": str(ObjectId())}  # ephemeral id for response consistency
        items_out.append(OrderItemOut(**i_out))

    return OrderOut(
        id=str(saved["_id"]),
        order_number=saved["order_number"],
        created_at=saved["created_at"],
        subtotal=saved["subtotal"],
        discount=saved["discount"],
        tax=saved["tax"],
        total=saved["total"],
        payment_method=saved["payment_method"],
        customer_id=saved.get("customer_id"),
        items=items_out,
    )


# -----------------------------
# Orders list/detail
# -----------------------------
@app.get("/orders", response_model=List[OrderOut])
def list_orders(limit: int = 50, offset: int = 0):
    cursor = db["order"].find({}).sort("created_at", -1).skip(offset).limit(limit)
    out: List[OrderOut] = []
    for o in cursor:
        items_out = []
        for i in o.get("items", []):
            i_out = {**i, "id": str(ObjectId())}
            items_out.append(OrderItemOut(**i_out))
        out.append(
            OrderOut(
                id=str(o["_id"]),
                order_number=o["order_number"],
                created_at=o["created_at"],
                subtotal=o["subtotal"],
                discount=o["discount"],
                tax=o["tax"],
                total=o["total"],
                payment_method=o["payment_method"],
                customer_id=o.get("customer_id"),
                items=items_out,
            )
        )
    return out


@app.get("/orders/{order_id}", response_model=OrderOut)
def get_order(order_id: str):
    _id = oid(order_id)
    if not _id:
        raise HTTPException(status_code=404, detail="Order not found")
    o = db["order"].find_one({"_id": _id})
    if not o:
        raise HTTPException(status_code=404, detail="Order not found")

    items_out = []
    for i in o.get("items", []):
        i_out = {**i, "id": str(ObjectId())}
        items_out.append(OrderItemOut(**i_out))

    return OrderOut(
        id=str(o["_id"]),
        order_number=o["order_number"],
        created_at=o["created_at"],
        subtotal=o["subtotal"],
        discount=o["discount"],
        tax=o["tax"],
        total=o["total"],
        payment_method=o["payment_method"],
        customer_id=o.get("customer_id"),
        items=items_out,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
