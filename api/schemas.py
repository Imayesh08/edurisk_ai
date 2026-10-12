"""
Pydantic schemas and request/response models for the EduRisk AI FastAPI service.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class StudentProfileInput(BaseModel):
    # School & Demographic
    school: str = Field(..., description="Student school ('GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)", example="GP")
    sex: str = Field(..., description="Student sex ('F' - female or 'M' - male)", example="F")
    age: int = Field(..., ge=15, le=25, description="Student age in years", example=17)
    address: str = Field(..., description="Student home address type ('U' - urban or 'R' - rural)", example="U")
    famsize: str = Field(..., description="Family size ('LE3' - less or equal to 3 or 'GT3' - greater than 3)", example="GT3")
    Pstatus: str = Field(..., description="Parent cohabitation status ('T' - living together or 'A' - apart)", example="T")
    
    # Parental Background
    Medu: int = Field(..., ge=0, le=4, description="Mother's education (0 - none, 1 - 4th grade, 2 - 5th-9th, 3 - secondary, 4 - higher)", example=3)
    Fedu: int = Field(..., ge=0, le=4, description="Father's education (0 - none, 1 - 4th grade, 2 - 5th-9th, 3 - secondary, 4 - higher)", example=2)
    Mjob: str = Field(..., description="Mother's job ('teacher', 'health', 'services', 'at_home', 'other')", example="services")
    Fjob: str = Field(..., description="Father's job ('teacher', 'health', 'services', 'at_home', 'other')", example="other")
    reason: str = Field(..., description="Reason to choose this school ('home', 'reputation', 'course', 'other')", example="reputation")
    guardian: str = Field(..., description="Student's guardian ('mother', 'father', 'other')", example="mother")
    
    # Academic Habits & Support
    traveltime: int = Field(..., ge=1, le=4, description="Home to school travel time (1 - <15 min, 2 - 15-30, 3 - 30-60, 4 - >60 min)", example=1)
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1 - <2 hours, 2 - 2-5 hours, 3 - 5-10 hours, 4 - >10 hours)", example=2)
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (0 to 4)", example=0)
    schoolsup: str = Field(..., description="Extra educational support ('yes' or 'no')", example="no")
    famsup: str = Field(..., description="Family educational support ('yes' or 'no')", example="yes")
    paid: str = Field(..., description="Extra paid classes within the course subject ('yes' or 'no')", example="no")
    activities: str = Field(..., description="Extra-curricular activities ('yes' or 'no')", example="yes")
    nursery: str = Field(..., description="Attended nursery school ('yes' or 'no')", example="yes")
    higher: str = Field(..., description="Wants to take higher education ('yes' or 'no')", example="yes")
    internet: str = Field(..., description="Internet access at home ('yes' or 'no')", example="yes")
    romantic: str = Field(..., description="With a romantic relationship ('yes' or 'no')", example="no")
    
    # Social & Lifestyle Factors
    famrel: int = Field(..., ge=1, le=5, description="Quality of family relationships (from 1 - very bad to 5 - excellent)", example=4)
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (from 1 - very low to 5 - very high)", example=3)
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (from 1 - very low to 5 - very high)", example=3)
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (from 1 - very low to 5 - very high)", example=1)
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (from 1 - very low to 5 - very high)", example=1)
    health: int = Field(..., ge=1, le=5, description="Current health status (from 1 - very bad to 5 - very good)", example=5)
    absences: int = Field(..., ge=0, le=93, description="Number of school absences", example=2)
    
    # Early Diagnostic Assessment
    G1: int = Field(..., ge=0, le=20, description="First period grade (0 to 20 scale)", example=12)


class RiskDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str
    abs_impact: float


class SinglePredictionResponse(BaseModel):
    prediction: int
    risk_status: str
    risk_level: str
    risk_probability: float
    applied_threshold: float
    confidence_assessment: str
    top_risk_drivers: List[RiskDriver]
    top_protective_factors: List[RiskDriver]
    evidence_based_interventions: List[str]
    epistemic_disclaimer: str


class BatchPredictionRequest(BaseModel):
    students: List[StudentProfileInput]
    threshold: Optional[float] = None


class BatchPredictionItem(BaseModel):
    student_index: int
    predicted_risk_probability: float
    predicted_academic_risk: str
    risk_tier: str


class BatchPredictionResponse(BaseModel):
    total_processed: int
    total_at_risk: int
    at_risk_percentage: float
    predictions: List[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    pipeline_ready: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    decision_threshold: float
    primary_metric: str
    test_roc_auc: float
    test_pr_auc: float
    test_recall_minority: float
    numerical_features: List[str]
    categorical_features: List[str]
