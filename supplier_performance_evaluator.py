import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, Text, END, messagebox, Toplevel
import os

class SupplierPerformanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Supplier Performance Evaluation")

        self.label = Label(master, text="Supplier Performance Evaluation")
        self.label.pack()

        self.run_button = Button(master, text="Run Performance Evaluation", command=self.run_performance_evaluation)
        self.run_button.pack()

        self.output_text = Text(master, height=15, width=100)
        self.output_text.pack()

        self.plot_button = Button(master, text="Show Decision Tree", command=self.show_decision_tree)
        self.plot_button.pack()

        self.accuracy_button = Button(master, text="Check Accuracy", command=self.show_accuracy_info)
        self.accuracy_button.pack()

        self.download_button = Button(master, text="Download Results", command=self.download_results)
        self.download_button.pack()

        self.info_button = Button(master, text="Info", command=self.show_info)
        self.info_button.pack()

        # Initialize these attributes to None; they'll be set after the model runs
        self.mae = None
        self.rmse = None
        self.results = None  # To store the results for downloading

    def run_performance_evaluation(self):
        # Load data from the Excel file
        filepath = 'C:/Users/Frank/Desktop/Supplier_Performance_Data.xlsx'
        df = pd.read_excel(filepath)
        
        # Separate features and target
        X = df.drop(['Supplier_Rating', 'Supplier_ID'], axis=1)  # Supplier Rating is the target
        y = df['Supplier_Rating']

        # Split data into training and testing (70% train, 30% test)
        split_index = int(len(df) * 0.7)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Initialize the Decision Tree Regressor
        regressor = DecisionTreeRegressor(random_state=42)

        # Train the model on the training data
        regressor.fit(X_train, y_train)

        # Make predictions on the entire dataset
        y_pred_all = regressor.predict(X)

        # Evaluate the model on the test set
        y_pred_test = regressor.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred_test)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Combine Supplier_ID with predictions and sort by rating in descending order
        supplier_ids = df['Supplier_ID'].tolist()
        self.results = sorted(zip(supplier_ids, y_pred_all), key=lambda x: x[1], reverse=True)

        # Display results in the text box with rankings
        self.output_text.delete(1.0, END)
        self.output_text.insert(END, f"Mean Absolute Error (Test Set): {self.mae:.2f}\n")
        self.output_text.insert(END, f"Root Mean Squared Error (Test Set): {self.rmse:.2f}\n")
        self.output_text.insert(END, "Ranked Supplier Ratings (All Data):\n")
        for rank, (supplier_id, rating) in enumerate(self.results, start=1):
            self.output_text.insert(END, f"{rank}. Supplier ID {supplier_id}: {rating:.1f}\n")

    def download_results(self):
        if self.results:
            try:
                # Create a DataFrame from the results
                df_results = pd.DataFrame(self.results, columns=['Supplier_ID', 'Predicted_Rating'])
                df_results['Rank'] = df_results.index + 1

                # Save the DataFrame to an Excel file
                output_path = 'C:/Users/Frank/Desktop/Supplier_Performance_Results.xlsx'
                df_results.to_excel(output_path, index=False)

                messagebox.showinfo("Success", f"Results successfully saved to {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please run the performance evaluation first.")

    def show_accuracy_info(self):
        if self.mae is not None and self.rmse is not None:
            # Determine the accuracy level
            if self.mae < 2 and self.rmse < 2:
                accuracy_level = "High"
            elif 2 <= self.mae <= 5 or 2 <= self.rmse <= 5:
                accuracy_level = "Medium"
            else:
                accuracy_level = "Low"

            accuracy_info = (
                f"Model Accuracy Metrics (Test Set):\n\n"
                f"Mean Absolute Error (MAE): {self.mae:.2f}\n"
                f"Root Mean Squared Error (RMSE): {self.rmse:.2f}\n\n"
                f"Accuracy Level: {accuracy_level}\n\n"
                f"Interpretation:\n"
                f"- The Mean Absolute Error (MAE) indicates that, on average, the predictions are "
                f"off by {self.mae:.2f} units from the actual supplier ratings.\n"
                f"- The Root Mean Squared Error (RMSE) indicates that the predictions have a "
                f"standard deviation of {self.rmse:.2f} units from the actual ratings.\n\n"
                f"A lower MAE and RMSE generally suggest a better model. These values help you "
                f"understand how well the model is performing in predicting supplier ratings."
            )
            messagebox.showinfo("Model Accuracy", accuracy_info)
        else:
            messagebox.showwarning("Warning", "Please run the performance evaluation first.")

    def show_info(self):
        # Text to be displayed in the information window
        info_text = (
            "The Supplier_Rating in your dataset is the actual score manually given based on past performance, "
            "while the Predicted Supplier Ratings are the model's estimates based on input features from the other columns. "
            "A high actual Supplier_Rating doesn't guarantee a high predicted rating, as the model might predict different "
            "values based on the patterns it has learned from the other columns."
        )
        
        # Display the information in a messagebox
        messagebox.showinfo("Information", info_text)

    def show_decision_tree(self):
        # Display the decision tree plot
        try:
            plt.figure(figsize=(20, 10))
            regressor = DecisionTreeRegressor(random_state=42)
            filepath = 'C:/Users/Frank/Desktop/Supplier_Performance_Data.xlsx'
            df = pd.read_excel(filepath)
            X = df.drop(['Supplier_Rating', 'Supplier_ID'], axis=1)
            y = df['Supplier_Rating']
            regressor.fit(X, y)
            tree.plot_tree(regressor, feature_names=X.columns, filled=True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run the Tkinter application
root = Tk()
app = SupplierPerformanceApp(root)
root.mainloop()
