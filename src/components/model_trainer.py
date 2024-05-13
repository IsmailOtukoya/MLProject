import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


#sys.path.append(os.path.abspath(r"C:\Users\USER\Documents\mlproject"))
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data",)
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME'),
            }
            params={
                "Random Forest Classifier": {
                 'n_estimators': [8, 16, 32, 64, 128, 256]
                 },
                "Gradient Boosting Classifier": {
                  'learning_rate': [.1, .01, .05, .001],
                  'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                 },
                "Logistic Regression": {
                # Define logistic regression specific hyperparameters here if any
                 },
                "CatBoosting Classifier": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
                 },
                "AdaBoost Classifier": {
                'learning_rate': [.1, .01, 0.5, .001],
                 'n_estimators': [8, 16, 32, 64, 128, 256]
                 }
                
            }

            # Call evaluate_models function to evaluate the models
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Determine the best model based on accuracy
            best_model_accuracy = max(sorted(model_report.values()))

            if best_model_accuracy < 0.6:  # Check if accuracy is below threshold
                raise CustomException("No best model found", {"reason": "Accuracy below threshold"})

            best_model_names = [model_name for model_name, accuracy in model_report.items() if
                                accuracy == best_model_accuracy]


            # Select the first model name with the highest accuracy
            best_model_name = best_model_names[0]  # Alternatively, you can choose a tie-breaking strategy here

            # Get the best model using the selected name
            best_model = models[best_model_name]

            # Log information about the best model found on both the training and testing datasets
            logging.info(f"Best found model based on accuracy: {best_model_name}. Accuracy: {best_model_accuracy}")

            # Save the best model to a file using the specified file path
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Make predictions on the test data using the best model
            predicted = best_model.predict(X_test)

            # Calculate the accuracy of the predictions compared to the actual test labels
            test_accuracy = accuracy_score(y_test, predicted)

            # Return the accuracy score
            return test_accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)