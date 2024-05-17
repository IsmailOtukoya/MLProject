import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(r"C:\Users\USER\Documents\mlproject"))
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 business_travel: str,
                 department: str,
                 education_field: str,
                 gender: str,
                 job_role: str,
                 marital_status: str,
                 age: int,
                 daily_rate: int,
                 distance_from_home: int,
                 education: int,
                 environment_satisfaction: int,
                 hourly_rate: int,
                 job_involvement: int,
                 job_level: int,
                 job_satisfaction: int,
                 monthly_income: int,
                 monthly_rate: int,
                 num_companies_worked: int,
                 over_time: str,
                 percent_salary_hike: int,
                 performance_rating: int,
                 relationship_satisfaction: int,
                 stock_option_level: int,
                 total_working_years: int,
                 training_times_last_year: int,
                 work_life_balance: int,
                 years_at_company: int,
                 years_in_current_role: int,
                 years_since_last_promotion: int,
                 years_with_curr_manager: int,
                 employee_number: int,
                 employee_count: int,
                 standard_hours: int,
                 over_18: str):

        self.business_travel = business_travel
        self.department = department
        self.education_field = education_field
        self.gender = gender
        self.job_role = job_role
        self.marital_status = marital_status
        self.age = age
        self.daily_rate = daily_rate
        self.distance_from_home = distance_from_home
        self.education = education
        self.environment_satisfaction = environment_satisfaction
        self.hourly_rate = hourly_rate
        self.job_involvement = job_involvement
        self.job_level = job_level
        self.job_satisfaction = job_satisfaction
        self.monthly_income = monthly_income
        self.monthly_rate = monthly_rate
        self.num_companies_worked = num_companies_worked
        self.over_time = over_time
        self.percent_salary_hike = percent_salary_hike
        self.performance_rating = performance_rating
        self.relationship_satisfaction = relationship_satisfaction
        self.stock_option_level = stock_option_level
        self.total_working_years = total_working_years
        self.training_times_last_year = training_times_last_year
        self.work_life_balance = work_life_balance
        self.years_at_company = years_at_company
        self.years_in_current_role = years_in_current_role
        self.years_since_last_promotion = years_since_last_promotion
        self.years_with_curr_manager = years_with_curr_manager
        self.employee_number = employee_number
        self.employee_count = employee_count
        self.standard_hours = standard_hours
        self.over_18 = over_18

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "BusinessTravel": [self.business_travel],
                "Department": [self.department],
                "EducationField": [self.education_field],
                "Gender": [self.gender],
                "JobRole": [self.job_role],
                "MaritalStatus": [self.marital_status],
                "Age": [self.age],
                "DailyRate": [self.daily_rate],
                "DistanceFromHome": [self.distance_from_home],
                "Education": [self.education],
                "EnvironmentSatisfaction": [self.environment_satisfaction],
                "HourlyRate": [self.hourly_rate],
                "JobInvolvement": [self.job_involvement],
                "JobLevel": [self.job_level],
                "JobSatisfaction": [self.job_satisfaction],
                "MonthlyIncome": [self.monthly_income],
                "MonthlyRate": [self.monthly_rate],
                "NumCompaniesWorked": [self.num_companies_worked],
                "OverTime": [self.over_time],
                "PercentSalaryHike": [self.percent_salary_hike],
                "PerformanceRating": [self.performance_rating],
                "RelationshipSatisfaction": [self.relationship_satisfaction],
                "StockOptionLevel": [self.stock_option_level],
                "TotalWorkingYears": [self.total_working_years],
                "TrainingTimesLastYear": [self.training_times_last_year],
                "WorkLifeBalance": [self.work_life_balance],
                "YearsAtCompany": [self.years_at_company],
                "YearsInCurrentRole": [self.years_in_current_role],
                "YearsSinceLastPromotion": [self.years_since_last_promotion],
                "YearsWithCurrManager": [self.years_with_curr_manager],
                "EmployeeNumber": [self.employee_number],
                "EmployeeCount": [self.employee_count],
                "StandardHours": [self.standard_hours],
                "Over18": [self.over_18]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
