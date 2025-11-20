import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    create_engine,
    Boolean,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# ---------------------------------------------
# Database configuration with safe cloud fallback
# ---------------------------------------------
# Priority:
# 1) DATABASE_URL if provided
# 2) FORCE_MYSQL=1 -> use MySQL envs
# 3) USE_SQLITE=1 (default) -> use SQLite local file (works in preview)
# 4) Otherwise fall back to SQLite

DATABASE_URL = os.getenv("DATABASE_URL")
FORCE_MYSQL = os.getenv("FORCE_MYSQL", "0") == "1"
USE_SQLITE = os.getenv("USE_SQLITE", "1") == "1"

if not DATABASE_URL:
    if FORCE_MYSQL and not USE_SQLITE:
        MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
        MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        MYSQL_DB = os.getenv("MYSQL_DB", "posdb")
        MYSQL_USER = os.getenv("MYSQL_USER", "root")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
        DATABASE_URL = (
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
        )
    else:
        # Default to SQLite for preview environment
        DATABASE_URL = "sqlite:///./pos.db"

# Create engine and session
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------
# SQLAlchemy Models
# ---------------------------------------------
class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    products = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    sku = Column(String(100), unique=True, nullable=True)
    barcode = Column(String(100), unique=True, nullable=True)
    price = Column(Float, nullable=False, default=0.0)
    cost = Column(Float, nullable=False, default=0.0)
    stock = Column(Float, nullable=False, default=0.0)
    tax_rate = Column(Float, nullable=False, default=0.0)  # percent e.g., 10 for 10%
    active = Column(Boolean, nullable=False, default=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    category = relationship("Category", back_populates="products")


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    phone = Column(String(100), nullable=True)
    email = Column(String(200), nullable=True)


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(50), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    subtotal = Column(Float, default=0.0, nullable=False)
    discount = Column(Float, default=0.0, nullable=False)
    tax = Column(Float, default=0.0, nullable=False)
    total = Column(Float, default=0.0, nullable=False)
    payment_method = Column(String(50), default="cash", nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)

    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    product_name = Column(String(200), nullable=False)
    qty = Column(Float, nullable=False, default=1.0)
    price = Column(Float, nullable=False, default=0.0)
    discount = Column(Float, nullable=False, default=0.0)
    tax = Column(Float, nullable=False, default=0.0)
    line_total = Column(Float, nullable=False, default=0.0)

    order = relationship("Order", back_populates="items")


class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(String(500), nullable=True)


# ---------------------------------------------
# Pydantic Schemas
# ---------------------------------------------
class CategoryIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class CategoryOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class ProductIn(BaseModel):
    name: str
    sku: Optional[str] = None
    barcode: Optional[str] = None
    price: float = 0.0
    cost: float = 0.0
    stock: float = 0.0
    tax_rate: float = 0.0
    active: bool = True
    category_id: Optional[int] = None


class ProductOut(BaseModel):
    id: int
    name: str
    sku: Optional[str]
    barcode: Optional[str]
    price: float
    cost: float
    stock: float
    tax_rate: float
    active: bool
    category_id: Optional[int]
    category_name: Optional[str] = None

    class Config:
        from_attributes = True


class CustomerIn(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None


class CustomerOut(BaseModel):
    id: int
    name: str
    phone: Optional[str]
    email: Optional[str]

    class Config:
        from_attributes = True


class POSItem(BaseModel):
    product_id: Optional[int] = None
    name: str
    qty: float
    price: float
    discount: float = 0.0  # absolute per line or per item (assume absolute total line discount)
    tax_rate: float = 0.0  # percent


class PricingRequest(BaseModel):
    items: List[POSItem]
    order_discount: float = 0.0  # absolute discount


class PricingResponse(BaseModel):
    subtotal: float
    discount: float
    tax: float
    total: float


class CheckoutRequest(BaseModel):
    items: List[POSItem]
    order_discount: float = 0.0
    payment_method: str = "cash"
    customer_id: Optional[int] = None


class OrderItemOut(BaseModel):
    id: int
    product_id: Optional[int]
    product_name: str
    qty: float
    price: float
    discount: float
    tax: float
    line_total: float

    class Config:
        from_attributes = True


class OrderOut(BaseModel):
    id: int
    order_number: str
    created_at: datetime
    subtotal: float
    discount: float
    tax: float
    total: float
    payment_method: str
    customer_id: Optional[int]
    items: List[OrderItemOut]

    class Config:
        from_attributes = True


# ---------------------------------------------
# FastAPI App
# ---------------------------------------------
app = FastAPI(title="POS API (Olsera-style)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get DB session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility: auto-create tables on startup
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------
# Health and Info
# ---------------------------------------------
@app.get("/")
def root():
    return {
        "message": "POS Backend Running",
        "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        "driver": "sqlite" if DATABASE_URL.startswith("sqlite") else "mysql/pymysql",
    }


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------
# Categories CRUD
# ---------------------------------------------
@app.get("/categories", response_model=List[CategoryOut])
def list_categories(db: Session = Depends(get_db)):
    return db.query(Category).order_by(Category.name).all()


@app.post("/categories", response_model=CategoryOut)
def create_category(payload: CategoryIn, db: Session = Depends(get_db)):
    exists = db.query(Category).filter(Category.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Category already exists")
    obj = Category(name=payload.name)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.put("/categories/{category_id}", response_model=CategoryOut)
def update_category(category_id: int, payload: CategoryIn, db: Session = Depends(get_db)):
    obj = db.query(Category).get(category_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Category not found")
    obj.name = payload.name
    db.commit()
    db.refresh(obj)
    return obj


@app.delete("/categories/{category_id}")
def delete_category(category_id: int, db: Session = Depends(get_db)):
    obj = db.query(Category).get(category_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Category not found")
    db.delete(obj)
    db.commit()
    return {"message": "deleted"}


# ---------------------------------------------
# Products CRUD + search
# ---------------------------------------------
@app.get("/products", response_model=List[ProductOut])
def list_products(
    q: Optional[str] = Query(None, description="Search by name, sku, barcode"),
    category_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Product)
    if q:
        like = f"%{q}%"
        query = query.filter(
            (Product.name.ilike(like))
            | (Product.sku.ilike(like))
            | (Product.barcode.ilike(like))
        )
    if category_id:
        query = query.filter(Product.category_id == category_id)
    items = query.order_by(Product.name).offset(offset).limit(limit).all()
    result: List[ProductOut] = []
    for p in items:
        result.append(
            ProductOut(
                id=p.id,
                name=p.name,
                sku=p.sku,
                barcode=p.barcode,
                price=p.price,
                cost=p.cost,
                stock=p.stock,
                tax_rate=p.tax_rate,
                active=p.active,
                category_id=p.category_id,
                category_name=p.category.name if p.category else None,
            )
        )
    return result


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    p = db.query(Product).get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return ProductOut(
        id=p.id,
        name=p.name,
        sku=p.sku,
        barcode=p.barcode,
        price=p.price,
        cost=p.cost,
        stock=p.stock,
        tax_rate=p.tax_rate,
        active=p.active,
        category_id=p.category_id,
        category_name=p.category.name if p.category else None,
    )


@app.post("/products", response_model=ProductOut)
def create_product(payload: ProductIn, db: Session = Depends(get_db)):
    if payload.sku:
        exists = db.query(Product).filter(Product.sku == payload.sku).first()
        if exists:
            raise HTTPException(status_code=400, detail="SKU already exists")
    if payload.barcode:
        exists = db.query(Product).filter(Product.barcode == payload.barcode).first()
        if exists:
            raise HTTPException(status_code=400, detail="Barcode already exists")
    obj = Product(**payload.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return get_product(obj.id, db)


@app.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: int, payload: ProductIn, db: Session = Depends(get_db)):
    obj: Optional[Product] = db.query(Product).get(product_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Product not found")
    # uniqueness checks
    if payload.sku and payload.sku != obj.sku:
        exists = db.query(Product).filter(Product.sku == payload.sku).first()
        if exists:
            raise HTTPException(status_code=400, detail="SKU already exists")
    if payload.barcode and payload.barcode != obj.barcode:
        exists = db.query(Product).filter(Product.barcode == payload.barcode).first()
        if exists:
            raise HTTPException(status_code=400, detail="Barcode already exists")

    for k, v in payload.model_dump().items():
        setattr(obj, k, v)
    db.commit()
    db.refresh(obj)
    return get_product(obj.id, db)


@app.delete("/products/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    obj = db.query(Product).get(product_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Product not found")
    db.delete(obj)
    db.commit()
    return {"message": "deleted"}


# ---------------------------------------------
# Customers CRUD
# ---------------------------------------------
@app.get("/customers", response_model=List[CustomerOut])
def list_customers(db: Session = Depends(get_db)):
    return db.query(Customer).order_by(Customer.name).all()


@app.post("/customers", response_model=CustomerOut)
def create_customer(payload: CustomerIn, db: Session = Depends(get_db)):
    obj = Customer(**payload.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.put("/customers/{customer_id}", response_model=CustomerOut)
def update_customer(customer_id: int, payload: CustomerIn, db: Session = Depends(get_db)):
    obj = db.query(Customer).get(customer_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Customer not found")
    for k, v in payload.model_dump().items():
        setattr(obj, k, v)
    db.commit()
    db.refresh(obj)
    return obj


@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    obj = db.query(Customer).get(customer_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Customer not found")
    db.delete(obj)
    db.commit()
    return {"message": "deleted"}


# ---------------------------------------------
# POS Pricing and Checkout
# ---------------------------------------------

def calculate_pricing(items: List[POSItem], order_discount: float) -> PricingResponse:
    subtotal = 0.0
    discount_total = 0.0
    tax_total = 0.0

    for it in items:
        line_subtotal = it.qty * it.price
        line_discount = min(it.discount, line_subtotal)
        taxable_base = max(line_subtotal - line_discount, 0.0)
        line_tax = taxable_base * (it.tax_rate / 100.0)
        # line_total not used directly here but kept for clarity
        # line_total = taxable_base + line_tax

        subtotal += line_subtotal
        discount_total += line_discount
        tax_total += line_tax

    # apply order-level discount (absolute)
    discount_total += min(order_discount, max(subtotal - discount_total, 0.0))

    total = max(subtotal - discount_total, 0.0) + tax_total

    # round to 2 decimals
    subtotal = round(subtotal, 2)
    discount_total = round(discount_total, 2)
    tax_total = round(tax_total, 2)
    total = round(total, 2)

    return PricingResponse(
        subtotal=subtotal,
        discount=discount_total,
        tax=tax_total,
        total=total,
    )


@app.post("/pos/pricing", response_model=PricingResponse)
def pos_pricing(payload: PricingRequest):
    return calculate_pricing(payload.items, payload.order_discount)


def generate_order_number(db: Session) -> str:
    # Simple YYYYMMDD-XXXX increment
    today = datetime.utcnow().strftime("%Y%m%d")
    like = f"{today}-%"
    last = (
        db.query(Order)
        .filter(Order.order_number.like(like))
        .order_by(Order.order_number.desc())
        .first()
    )
    if not last:
        return f"{today}-0001"
    try:
        seq = int(last.order_number.split("-")[1]) + 1
    except Exception:
        seq = 1
    return f"{today}-{seq:04d}"


@app.post("/pos/checkout", response_model=OrderOut)
def pos_checkout(payload: CheckoutRequest, db: Session = Depends(get_db)):
    # Validate products and stocks
    for it in payload.items:
        if it.product_id:
            p = db.query(Product).get(it.product_id)
            if not p:
                raise HTTPException(status_code=400, detail=f"Product {it.product_id} not found")
            if p.stock is not None and p.stock < it.qty:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for {p.name}")

    pricing = calculate_pricing(payload.items, payload.order_discount)

    order = Order(
        order_number=generate_order_number(db),
        subtotal=pricing.subtotal,
        discount=pricing.discount,
        tax=pricing.tax,
        total=pricing.total,
        payment_method=payload.payment_method,
        customer_id=payload.customer_id,
    )
    db.add(order)
    db.flush()  # get order.id

    # Create items and adjust stock
    for it in payload.items:
        line_subtotal = it.qty * it.price
        line_discount = min(it.discount, line_subtotal)
        taxable_base = max(line_subtotal - line_discount, 0.0)
        line_tax = taxable_base * (it.tax_rate / 100.0)
        line_total = taxable_base + line_tax

        oi = OrderItem(
            order_id=order.id,
            product_id=it.product_id,
            product_name=it.name,
            qty=it.qty,
            price=it.price,
            discount=round(line_discount, 2),
            tax=round(line_tax, 2),
            line_total=round(line_total, 2),
        )
        db.add(oi)

        if it.product_id:
            p = db.query(Product).get(it.product_id)
            if p:
                p.stock = round((p.stock or 0.0) - it.qty, 4)
                db.add(p)

    db.commit()
    db.refresh(order)

    # Compose response
    items_out = [
        OrderItemOut(
            id=i.id,
            product_id=i.product_id,
            product_name=i.product_name,
            qty=i.qty,
            price=i.price,
            discount=i.discount,
            tax=i.tax,
            line_total=i.line_total,
        )
        for i in order.items
    ]

    return OrderOut(
        id=order.id,
        order_number=order.order_number,
        created_at=order.created_at,
        subtotal=order.subtotal,
        discount=order.discount,
        tax=order.tax,
        total=order.total,
        payment_method=order.payment_method,
        customer_id=order.customer_id,
        items=items_out,
    )


# ---------------------------------------------
# Orders list/detail
# ---------------------------------------------
@app.get("/orders", response_model=List[OrderOut])
def list_orders(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    orders = (
        db.query(Order)
        .order_by(Order.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    result: List[OrderOut] = []
    for o in orders:
        items_out = [
            OrderItemOut(
                id=i.id,
                product_id=i.product_id,
                product_name=i.product_name,
                qty=i.qty,
                price=i.price,
                discount=i.discount,
                tax=i.tax,
                line_total=i.line_total,
            )
            for i in o.items
        ]
        result.append(
            OrderOut(
                id=o.id,
                order_number=o.order_number,
                created_at=o.created_at,
                subtotal=o.subtotal,
                discount=o.discount,
                tax=o.tax,
                total=o.total,
                payment_method=o.payment_method,
                customer_id=o.customer_id,
                items=items_out,
            )
        )
    return result


@app.get("/orders/{order_id}", response_model=OrderOut)
def get_order(order_id: int, db: Session = Depends(get_db)):
    o = db.query(Order).get(order_id)
    if not o:
        raise HTTPException(status_code=404, detail="Order not found")
    items_out = [
        OrderItemOut(
            id=i.id,
            product_id=i.product_id,
            product_name=i.product_name,
            qty=i.qty,
            price=i.price,
            discount=i.discount,
            tax=i.tax,
            line_total=i.line_total,
        )
        for i in o.items
    ]
    return OrderOut(
        id=o.id,
        order_number=o.order_number,
        created_at=o.created_at,
        subtotal=o.subtotal,
        discount=o.discount,
        tax=o.tax,
        total=o.total,
        payment_method=o.payment_method,
        customer_id=o.customer_id,
        items=items_out,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
