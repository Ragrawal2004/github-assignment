import pandera.pandas as pa
from pandera import Field


class LoanSchema(pa.DataFrameModel):

    person_age: float = Field(ge=18)

    person_gender: str

    person_education: str

    person_income: float = Field(gt=0)

    person_emp_exp: int = Field(ge=0)

    person_home_ownership: str

    loan_amnt: float = Field(gt=0)

    loan_intent: str

    loan_int_rate: float = Field(gt=0)

    loan_percent_income: float = Field(ge=0)

    cb_person_cred_hist_length: float = Field(ge=0)

    credit_score: int = Field(ge=300, le=850)

    previous_loan_defaults_on_file: str

    loan_status: int = Field(isin=[0, 1])

    class Config:
        strict = True
