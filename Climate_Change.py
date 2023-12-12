#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploring and analyzing World Bank data related to climate change indicators.

This script reads and analyzes data on CO2 emissions and access to electricity
from World Bank datasets. It performs data cleaning, reshaping, and 
visualization.

"""

# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()

import warnings
warnings.filterwarnings("ignore")

# Function for dropping columns

def drop_empty_columns(dataset):
    """
    Drop columns where all values are NaN.

    Parameters:
    - dataset (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with empty columns dropped.
    """
    dataset_cleaned = dataset.dropna(axis=1, how='all')
    return dataset_cleaned

# Function for dropping rows

def drop_rows(dataset,row_names):
    """
    Drop rows based on specified row names.

    Parameters:
    - dataset (pd.DataFrame): Input DataFrame.
    - row_names (list): List of row names to be dropped.

    Returns:
    - pd.DataFrame: DataFrame with specified rows dropped.
    """
    dataset = dataset.drop(row_names,axis = 0)
    return dataset


def read_and_clean_world_bank_data(filename):
    """
   Read and clean World Bank data from a CSV file.

   Parameters:
   - filename (str): Path to the CSV file.

   Returns:
   - pd.DataFrame: Cleaned and transposed DataFrame.
   """
    # Read the data
    df = pd.read_csv(filename, skiprows=4)

    # Clean the data by dropping empty columns
    df_cleaned = drop_empty_columns(df)

    # Transpose the dataframe
    df_transposed = df_cleaned.transpose().reset_index()

    # Set column names to the first row
    df_transposed.columns = df_transposed.iloc[0]

    # Drop the first row (which is now redundant)
    df_transposed = df_transposed.drop(0)

    # Rename the columns for better readability
    df_transposed = df_transposed.rename\
    (columns={'Country Name': 'Year'}).reset_index(drop=True)

    return df_transposed


co2_emissions_df= pd.read_csv("CO2emissions.csv", skiprows=4)
co2_emissions_df = drop_empty_columns(co2_emissions_df)
co2_emissions_df.head()

access_to_electricity_df= pd.read_csv("Electricity.csv", skiprows=4)
access_to_electricity_df = drop_empty_columns(access_to_electricity_df)
access_to_electricity_df.head()


# Read and clean CO2 emissions data
co2_emissions_df2 = read_and_clean_world_bank_data("CO2emissions.csv")
co2_emissions_df2.head()

# Read and clean access to electricity data
access_to_electricity_df2 = read_and_clean_world_bank_data("Electricity.csv")
access_to_electricity_df2.head()

def reshape_world_bank_data(filename):
    """
    Reshape World Bank data by melting the DataFrame.

    Parameters:
    - filename (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Reshaped DataFrame.
    """
    # Read the data
    df = pd.read_csv(filename, skiprows=4)

    # Clean the data by dropping empty columns
    df_cleaned = drop_empty_columns(df)

    # Melt the DataFrame to reshape it
    df_reshaped = pd.melt(df_cleaned, id_vars=['Country Name', 
                    'Country Code', 'Indicator Name', 'Indicator Code'],
       var_name='Year', value_name='Value')

    return df_reshaped

# Reshape CO2 emissions data
co2_emissions_df_reshaped = reshape_world_bank_data("CO2emissions.csv")
co2_emissions_df_reshaped.head()
# Reshape access to electricity data
access_to_electricity_df_reshaped = reshape_world_bank_data("Electricity.csv")
access_to_electricity_df_reshaped.head()


# Merge dataframes on 'Country Name' and 'Year'
merged_df = pd.merge(co2_emissions_df_reshaped, 
                     access_to_electricity_df_reshaped, 
             on=['Country Name', 'Year'], suffixes=('_co2', '_electricity'))

# Drop rows with NaN values
merged_df = merged_df.dropna()

# Display the merged and cleaned dataframe
print(merged_df.head())



# Select a few countries of interest
selected_countries = ['United States', 'China', 'India', 'Brazil']

# Filter the merged dataframe for the selected countries
selected_df = merged_df[merged_df['Country Name'].isin(selected_countries)]

# Display summary statistics using .describe()
summary_stats = selected_df.describe()
print("Summary Statistics:")
print(summary_stats)

# Convert 'Value_co2' and 'Value_electricity' to numeric data types
selected_df['Value_co2'] = pd.to_numeric(selected_df['Value_co2'])
selected_df['Value_electricity'] = pd.to_numeric
(selected_df['Value_electricity'])
# Convert 'Year' column to numeric
selected_df['Year'] = pd.to_numeric(selected_df['Year'], errors='coerce')

# Convert 'Value_co2' and 'Value_electricity' columns to numeric
selected_df[['Value_co2', 'Value_electricity']] = selected_df\
    [['Value_co2', 'Value_electricity']].apply(pd.to_numeric, errors='coerce')
print(selected_df.dtypes)

# Now, calculate mean and median, excluding non-numeric columns
mean_values = selected_df.groupby('Country Name')\
               [['Value_co2', 'Value_electricity']].mean()
median_values = selected_df.groupby('Country Name')\
                 [['Value_co2', 'Value_electricity']].median()

# Display the results
print("Summary Statistics:")
print(summary_stats)

print("\n Mean Values:")
print(mean_values)

print("\nMedian Values:")
print(median_values)




import seaborn as sns
import matplotlib.pyplot as plt

# Filter data for China
china_data = selected_df[selected_df['Country Name'] == 'China']

# Select relevant columns for the heatmap
heatmap_data_china = china_data[['Value_co2', 'Value_electricity']]

# Create a correlation matrix for China
correlation_matrix_china = heatmap_data_china.corr()

# Print the correlation value
correlation_value_china = correlation_matrix_china.loc\
                            ['Value_co2', 'Value_electricity']
print(f'Correlation Value for China: {correlation_value_china}')

# Create a heatmap for China
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_china, annot=True, cmap='coolwarm',
                                             fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for CO2 Emissions and \
          Access to Electricity in China')
plt.show()


from scipy.stats import linregress

# Scatter plot with regression line
plt.figure(figsize=(12, 6))
sns.regplot(x='Value_electricity', y='Value_co2', 
            data=selected_df, scatter_kws={'s': 50}, line_kws={'color': 'red'})

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress\
                   (selected_df['Value_electricity'], selected_df['Value_co2'])

# Plot the regression line
x_values = np.linspace(selected_df['Value_electricity'].min(), 
                       selected_df['Value_electricity'].max(), 100)
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, color='blue', 
         label=f'Regression Line (R-squared = {r_value**2:.2f})')

plt.title('Relationship between Access to Electricity and CO2 Emissions')
plt.xlabel('Access to Electricity (% of Population)')
plt.ylabel('CO2 Emissions (kt)')
plt.legend()
plt.show()


# Select relevant columns for the heatmap
heatmap_data = selected_df[['Value_co2', 'Value_electricity']]

# Create a correlation matrix for the selected columns
heatmap_correlation = heatmap_data.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_correlation, annot=True, 
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap: CO2 Emissions and Access to Electricity')
plt.show()


# Correlation matrix
correlation_matrix = selected_df[['Value_co2', 'Value_electricity']].corr()
print("Correlation Matrix:")
print(correlation_matrix)


# Filter data for the selected countries and years (2015-2020)
selected_countries_years_data = selected_df[(selected_df['Country Name'].isin
                                 (selected_countries)) & 
             (selected_df['Year'] >= 2015) & (selected_df['Year'] <= 2021)]

# Bar graph for average CO2 emissions and access to electricity by country
plt.figure(figsize=(12, 8))
sns.barplot(x='Country Name', y='Value_electricity', 
            data=selected_countries_years_data, color='green', 
            label='Access to Electricity')
plt.title('Comparison of CO2 Emissions and Access to \
          Electricity for Selected Countries (2015-2021)')
plt.xlabel('Country')
plt.ylabel('Average Value')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()


# Bar graph for average CO2 emissions and access to electricity by country
plt.figure(figsize=(12, 8))
sns.barplot(x='Country Name', y='Value_co2', 
     data=selected_countries_years_data, color='blue', label='CO2 Emissions')
plt.title('Comparison of CO2 Emissions for Selected Countries (2015-2021)')
plt.xlabel('Country')
plt.ylabel('Average Value')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()
