"""
Database Schemas for POS App (MongoDB)

Each Pydantic model represents a collection in MongoDB. Collection name is the
lowercase of the class name by convention.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Core Master Data
class Category(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class Product(BaseModel):
    name: str
    sku: Optional[str] = None
    barcode: Optional[str] = None
    price: float = 0.0
    cost: float = 0.0
    stock: float = 0.0
    tax_rate: float = 0.0
    active: bool = True
    category_id: Optional[str] = None  # ObjectId string

class Customer(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None

# POS / Orders
class OrderItem(BaseModel):
    product_id: Optional[str] = None
    product_name: str
    qty: float
    price: float
    discount: float = 0.0
    tax: float = 0.0
    line_total: float = 0.0

class Order(BaseModel):
    order_number: str
    created_at: datetime
    subtotal: float
    discount: float
    tax: float
    total: float
    payment_method: str = "cash"
    customer_id: Optional[str] = None
    items: List[OrderItem] = []

# Settings
class Setting(BaseModel):
    key: str
    value: Optional[str] = None
