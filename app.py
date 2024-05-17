from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application 

#Route for our home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            business_travel=request.form.get('business_travel'),
            department=request.form.get('department'),
            education_field=request.form.get('education_field'),
            gender=request.form.get('gender'),
            job_role=request.form.get('job_role'),
            marital_status=request.form.get('marital_status'),
            age=int(request.form.get('age')),
            daily_rate=int(request.form.get('daily_rate')),
            distance_from_home=int(request.form.get('distance_from_home')),
            education=int(request.form.get('education')),
            environment_satisfaction=int(request.form.get('environment_satisfaction')),
            hourly_rate=int(request.form.get('hourly_rate')),
            job_involvement=int(request.form.get('job_involvement')),
            job_level=int(request.form.get('job_level')),
            job_satisfaction=int(request.form.get('job_satisfaction')),
            monthly_income=int(request.form.get('monthly_income')),
            monthly_rate=int(request.form.get('monthly_rate')),
            num_companies_worked=int(request.form.get('num_companies_worked')),
            over_time=request.form.get('over_time'),
            percent_salary_hike=int(request.form.get('percent_salary_hike')),
            performance_rating=int(request.form.get('performance_rating')),
            relationship_satisfaction=int(request.form.get('relationship_satisfaction')),
            stock_option_level=int(request.form.get('stock_option_level')),
            total_working_years=int(request.form.get('total_working_years')),
            training_times_last_year=int(request.form.get('training_times_last_year')),
            work_life_balance=int(request.form.get('work_life_balance')),
            years_at_company=int(request.form.get('years_at_company')),
            years_in_current_role=int(request.form.get('years_in_current_role')),
            years_since_last_promotion=int(request.form.get('years_since_last_promotion')),
            years_with_curr_manager=int(request.form.get('years_with_curr_manager')),
            employee_number = int(request.form.get('employee_number')),
            employee_count = int(request.form.get('employee_count')),
            standard_hours = int(request.form.get('standard_hours')),
            over_18 = request.form.get('over_18')



        )
        pred_df=data.get_data_as_data_frame() #Would convert the whole dataset above to a dataframe
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__ =="__main__":
    app.run(host='0.0.0.0', debug=True)