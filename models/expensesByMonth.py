from pydantic import BaseModel

class ExpensesByMonth(BaseModel):
    month: str
    amount: float


