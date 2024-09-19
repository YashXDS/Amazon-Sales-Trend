# Amazon Sales Data Analysis and Prediction

## Project Overview

This project aims to perform comprehensive exploratory data analysis (EDA) and build a linear regression model to predict **Total Revenue** based on sales data. The dataset includes features such as **Units Sold**, **Unit Price**, **Unit Cost**, and various date-related fields. The project also includes detailed data visualization and statistical analysis to extract insights from the sales trends.

## Dataset

The dataset used for this analysis is a CSV file named `Amazon Sales data.csv`. It includes the following key columns:

- **Order Date**: Date when the order was placed.
- **Ship Date**: Date when the order was shipped.
- **Order Priority**: Priority of the order.
- **Units Sold**: Number of units sold.
- **Unit Price**: Price of each unit sold.
- **Unit Cost**: Cost of each unit.
- **Total Revenue**: Total revenue generated from the order.
- **Total Cost**: Total cost of the order.
- **Total Profit**: Total profit generated from the order.
- **Sales Channel**: Channel through which the sale occurred (Online/Offline).

## Project Structure

1. **Data Cleaning and Preprocessing**:
   - Checked for missing and duplicate values.
   - Standardized date formats.
   - Extracted year, month, and year-month features from `Order Date` and `Ship Date`.
   - Saved the cleaned data to a new CSV file: `cleaned_data.csv`.

2. **Exploratory Data Analysis (EDA)**:
   - Calculated and visualized sales trends by month, year, and year-month.
   - Created distribution plots for numerical features such as `Units Sold`, `Unit Price`, `Total Revenue`, etc.
   - Plotted boxplots and correlation heatmaps to analyze relationships between features.

3. **Statistical Analysis**:
   - Calculated summary statistics (mean, median, standard deviation) for `Total Revenue`.
   - Performed a t-test to compare `Total Revenue` between sales channels (Online vs. Offline).
   - Conducted an ANOVA test to analyze the impact of `Order Priority` on `Total Revenue`.

4. **Machine Learning Model**:
   - Defined a linear regression model to predict `Total Revenue` based on `Units Sold`, `Unit Price`, and `Unit Cost`.
   - Split the dataset into training and testing sets (80% training, 20% testing).
   - Evaluated the model using metrics such as **Mean Squared Error (MSE)** and **R-squared**.
   - Provided interpretation of the model's coefficients and their impact on **Total Revenue**.

## Dependencies

To run this project, you will need the following Python libraries:

- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn

Install them using the following command:

```bash
pip install pandas matplotlib seaborn scipy scikit-learn
```
## How to Run

Clone this repository.
Place the dataset Amazon Sales data.csv in the same directory as the code.
Run the Python script to perform data cleaning, analysis, and build the machine learning model.
The results will include various visualizations and predictions for Total Revenue.

## Results and Insights

- **Sales Trend Analysis**: Monthly and yearly sales trends indicate seasonal peaks in sales.
- **Statistical Tests**: A t-test revealed differences in revenue between online and offline channels. An ANOVA test highlighted the effect of order priority on revenue.
- **Model Evaluation**: The linear regression model showed an R-squared score indicating the strength of the relationship between features and Total Revenue.

## Example Output

Predicted revenue for 1000 units sold with a unit price of $50 and a unit cost of $30:

``` bash
Predicted Revenue: $48,500
```
## Future Improvements

Implement more advanced machine learning models such as RandomForest or GradientBoosting to improve prediction accuracy.
Add more detailed analysis on different sales channels and product categories.
