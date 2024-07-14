import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load csv file into a pandas DataFrame
sales_data = pd.read_csv('Amazon Sales data.csv', )

# Display the first five rows of the DataFrame to inspect the data
print(sales_data.head())

# Check for missing values
print(sales_data.isnull().sum())                         # there is no missing values

# Check for duplicate rows
print(sales_data.duplicated().sum())                     # there is no duplicate rows

# Standardize date format 
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])
sales_data['Ship Date'] = pd.to_datetime(sales_data['Ship Date'])

# Save cleaned data to a new CSV file 
sales_data.to_csv('cleaned_data.csv', index=False)

# Extract year, month, and year-month features for Order Date
sales_data['Order Year'] = sales_data['Order Date'].dt.year
sales_data['Order Month'] = sales_data['Order Date'].dt.month
sales_data['Order Year-Month'] = sales_data['Order Date'].dt.to_period('M')
print(sales_data['Order Year'].head() )
print(sales_data['Order Month'].head())
print(sales_data['Order Year-Month'].head())



# Extract year, month, and year-month features for Ship Date
sales_data['Ship Year'] = sales_data['Ship Date'].dt.year
sales_data['Ship Month'] = sales_data['Ship Date'].dt.month
sales_data['Ship Year-Month'] = sales_data['Ship Date'].dt.to_period('M')

# Calculate total sales for each Order Month
order_month_sales = sales_data.groupby('Order Month')['Total Revenue'].sum().reset_index()

# Calculate total sales for each Order Year
order_year_sales = sales_data.groupby('Order Year')['Total Revenue'].sum().reset_index()

# Calculate total sales for each Order Year-Month combination
order_year_month_sales = sales_data.groupby('Order Year-Month')['Total Revenue'].sum().reset_index()

# Plotting the Order Month-wise sales trend
plt.figure(figsize=(10, 6))
plt.plot(order_month_sales['Order Month'], order_month_sales['Total Revenue'], marker='o')
plt.xlabel('Order Month')
plt.ylabel('Total Revenue')
plt.title('Order Month-wise Sales Trend')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

# Plotting the Order Year-wise sales trend
plt.figure(figsize=(10, 6))
plt.plot(order_year_sales['Order Year'], order_year_sales['Total Revenue'], marker='o')
plt.xlabel('Order Year')
plt.ylabel('Total Revenue')
plt.title('Order Year-wise Sales Trend')
plt.xticks(range(2010, 2018), [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017])
plt.grid(True)
plt.show()

# Plotting the Order Yearly Month-wise sales trend
order_year_month_str = order_year_month_sales['Order Year-Month'].astype(str)          # Convert 'Order Year-Month' values to strings for plotting
plt.figure(figsize=(10, 6))
plt.plot(order_year_month_str, order_year_month_sales['Total Revenue'], marker='o')
plt.xlabel('Order Year-Month')
plt.ylabel('Total Revenue')
plt.title('Order Yearly Month-wise Sales Trend')
plt.xticks(range(len(order_year_month_str)), order_year_month_str, rotation=90)        # Set the range and labels for x-axis ticks
plt.grid(True)
plt.show()

# Summary statistics
summary_stats = sales_data.describe()
print(summary_stats)

# Distribution of numerical features
num_cols = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(sales_data[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Boxplot for Total Revenue by Order Priority
plt.figure(figsize=(10, 6))
sns.boxplot(x='Order Priority', y='Total Revenue', data=sales_data)
plt.title('Boxplot of Total Revenue by Order Priority')
plt.show()

# Correlation heatmap for numerical features
corr_matrix = sales_data[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Monthly sales trend
monthly_sales = sales_data.groupby('Order Month')['Total Revenue'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['Order Month'], monthly_sales['Total Revenue'], marker='o')
plt.xlabel('Order Month')
plt.ylabel('Total Revenue')
plt.title('Monthly Sales Trend')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

# Calculate mean, median, and standard deviation of Total Revenue
mean_revenue = sales_data['Total Revenue'].mean()
median_revenue = sales_data['Total Revenue'].median()
std_revenue = sales_data['Total Revenue'].std()

print(f"Mean Total Revenue: {mean_revenue}")
print(f"Median Total Revenue: {median_revenue}")
print(f"Standard Deviation of Total Revenue: {std_revenue}")

# Perform t-test for Total Revenue between two sales channels (e.g., Online vs. Offline)
online_revenue = sales_data[sales_data['Sales Channel'] == 'Online']['Total Revenue']
offline_revenue = sales_data[sales_data['Sales Channel'] == 'Offline']['Total Revenue']

t_stat, p_value = stats.ttest_ind(online_revenue, offline_revenue, equal_var=False)
print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p_value}")

# Calculate correlation coefficients between Total Revenue and other numerical features
correlation_with_revenue = sales_data[num_cols].corr()['Total Revenue']
print("Correlation with Total Revenue:")
print(correlation_with_revenue)

# Perform ANOVA to test the impact of Order Priority on Total Revenue
anova_results = stats.f_oneway(*[group['Total Revenue'] for name, group in sales_data.groupby('Order Priority')])
print("ANOVA Results:")
print(anova_results)

# Define features (X) and target variable (y)
features = ['Units Sold', 'Unit Price', 'Unit Cost']
X = sales_data[features]
y = sales_data['Total Revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Example prediction for a new data point
new_data_point = [[1000, 50, 30]]  # Units Sold, Unit Price, Unit Cost
predicted_revenue = model.predict(new_data_point)
print(f"Predicted Revenue for the new data point: {predicted_revenue}")

# Print model coefficients (coefficients represent the impact of each feature on Total Revenue)
print("Model Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")

# Interpretation and Insights
print("\nInterpretation and Insights:")
print("- The coefficient for 'Units Sold' suggests that for every unit increase in sales, Total Revenue increases by approximately the 'Units Sold' coefficient value.")
print("- The coefficients for 'Unit Price' and 'Unit Cost' indicate the impact of pricing on Total Revenue.")
print("- A higher 'Unit Price' coefficient implies that increasing prices leads to higher Total Revenue, while a lower 'Unit Cost' coefficient suggests cost optimization strategies can improve Total Revenue.")

# Additional insights based on model evaluation
print("\nAdditional Insights based on Model Evaluation:")
print("- The Mean Squared Error (MSE) and R-squared score provide an indication of the model's predictive performance.")
print("- A lower MSE indicates better model accuracy, while a higher R-squared score (closer to 1) indicates a better fit of the model to the data.")

