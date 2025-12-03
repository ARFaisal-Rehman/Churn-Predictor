# data_generator.py - Generate sample data for testing the churn prediction dashboard

import pandas as pd
import numpy as np
import os

def generate_customer_data(n_samples=100, output_file='sample_customer_data.csv'):
    """Generate synthetic telecom customer data for demonstration
    
    Args:
        n_samples (int): Number of customer records to generate
        output_file (str): Filename to save the generated data
        
    Returns:
        str: Path to the generated file
    """
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:04d}' for i in range(1, n_samples + 1)]
    
    # Generate customer names (fictional)
    first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'Robert', 'Lisa', 
                  'William', 'Emma', 'James', 'Olivia', 'Daniel', 'Sophia', 'Matthew', 'Ava']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia',
                 'Rodriguez', 'Wilson', 'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Moore', 'Lee']
    
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_samples)]
    
    # Generate demographic features
    gender = np.random.choice(['Male', 'Female'], n_samples)
    age = np.random.randint(18, 85, n_samples)
    
    # Generate service features
    tenure = np.random.randint(1, 73, n_samples)  # 1-72 months
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    
    # Generate service types
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2])
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2])
    
    # Generate additional services (only for those with internet)
    has_internet = internet_service != 'No'
    online_security = np.where(has_internet, np.random.choice(['Yes', 'No'], n_samples), 'No internet service')
    online_backup = np.where(has_internet, np.random.choice(['Yes', 'No'], n_samples), 'No internet service')
    device_protection = np.where(has_internet, np.random.choice(['Yes', 'No'], n_samples), 'No internet service')
    tech_support = np.where(has_internet, np.random.choice(['Yes', 'No'], n_samples), 'No internet service')
    streaming_tv = np.where(has_internet, np.random.choice(['Yes', 'No'], n_samples), 'No internet service')
    streaming_movies = np.where(has_internet, np.random.choice(['Yes', 'No'], n_samples), 'No internet service')
    
    # Generate multiple lines (only for those with phone service)
    has_phone = phone_service == 'Yes'
    multiple_lines = np.where(has_phone, np.random.choice(['Yes', 'No'], n_samples), 'No phone service')
    
    # Generate financial features
    # Base charges
    base_internet_charge = np.where(internet_service == 'No', 0, 
                                  np.where(internet_service == 'DSL', 25, 50))
    base_phone_charge = np.where(phone_service == 'No', 0, 20)
    
    # Additional services charges
    service_charges = np.zeros(n_samples)
    for service in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        service_charges += np.where(service == 'Yes', np.random.uniform(5, 10, n_samples), 0)
    
    # Contract discounts
    contract_factor = np.where(contract == 'Month-to-month', 1.0, 
                              np.where(contract == 'One year', 0.9, 0.8))
    
    # Calculate monthly charges
    monthly_charges = (base_internet_charge + base_phone_charge + service_charges) * contract_factor
    monthly_charges = np.round(monthly_charges, 2)
    
    # Calculate total charges (with some variation)
    total_charges = monthly_charges * tenure * (np.random.uniform(0.95, 1.05, n_samples))
    total_charges = np.round(total_charges, 2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'name': names,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges
    })
    
    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_samples} customer records and saved to {output_path}")
    return output_path

def generate_historical_data(customer_df, months=12, output_file='sample_historical_data.csv'):
    """Generate historical service data for existing customers
    
    Args:
        customer_df (DataFrame or str): Customer dataframe or path to customer CSV file
        months (int): Number of months of historical data to generate
        output_file (str): Filename to save the generated data
        
    Returns:
        str: Path to the generated file
    """
    # Load customer data if string path provided
    if isinstance(customer_df, str):
        customer_df = pd.read_csv(customer_df)
    
    # List to store all historical records
    history_records = []
    
    # Current date (as reference point)
    current_date = pd.Timestamp.now()
    
    # For each customer
    for _, customer in customer_df.iterrows():
        customer_id = customer['customer_id']
        current_monthly_charge = customer['monthly_charges']
        
        # Generate history for each month
        for month_offset in range(months):
            # Calculate date (going backwards from current date)
            record_date = current_date - pd.DateOffset(months=month_offset)
            
            # Slightly vary the monthly charge to simulate changes
            if month_offset > 0:  # Keep current month charge as is
                monthly_charge = current_monthly_charge * np.random.uniform(0.9, 1.1)
                monthly_charge = round(monthly_charge, 2)
            else:
                monthly_charge = current_monthly_charge
            
            # Generate usage metrics
            data_usage_gb = np.random.uniform(10, 500) if customer['internet_service'] != 'No' else 0
            call_minutes = np.random.uniform(50, 1000) if customer['phone_service'] == 'Yes' else 0
            support_calls = np.random.randint(0, 4)  # 0-3 support calls per month
            
            # Payment status
            payment_status = np.random.choice(['Paid on time', 'Late payment', 'Missed payment'], 
                                            p=[0.85, 0.1, 0.05])
            
            # Create record
            record = {
                'customer_id': customer_id,
                'date': record_date.strftime('%Y-%m-%d'),
                'monthly_charge': monthly_charge,
                'data_usage_gb': round(data_usage_gb, 1),
                'call_minutes': round(call_minutes, 1),
                'support_calls': support_calls,
                'payment_status': payment_status
            }
            
            history_records.append(record)
    
    # Create DataFrame from all records
    history_df = pd.DataFrame(history_records)
    
    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    history_df.to_csv(output_path, index=False)
    
    print(f"Generated {len(history_records)} historical records and saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Generate customer data
    customer_file = generate_customer_data(n_samples=200)
    
    # Generate historical data based on the customer data
    generate_historical_data(customer_file, months=12)
    
    print("\nSample data generation complete! You can now use these files to test the Risk Radar dashboard.")
    print("Files generated:")
    print("1. sample_customer_data.csv - Customer profiles with demographic and service information")
    print("2. sample_historical_data.csv - 12 months of service history for each customer")