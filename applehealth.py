"""
Apple Health Data Analyzer
-------------------------

This script analyzes exported Apple Health data (export.xml) with a focus on:
- Steps
- Walking/Running Distance
- Heart Rate

"""

import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import openai
import os
from dotenv import load_dotenv
import plotly.express as px  # Add Plotly for interactive charts

def parse_health_data(file_path, record_type):
    """
    Parse specific health metrics from Apple Health export.xml file.
    
    Args:
        file_path (str): Path to the export.xml file
        record_type (str): The type of health record to parse (e.g., 'HKQuantityTypeIdentifierStepCount')
    
    Returns:
        pandas.DataFrame: DataFrame containing dates and values for the specified metric
    """
    print(f"Starting to parse {record_type}...")
    dates = []
    values = []
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    print("XML file loaded, searching records...")
    
    for record in root.findall('.//Record'):
        if record.get('type') == record_type:
            try:
                value = float(record.get('value'))
                date = datetime.strptime(record.get('endDate'), '%Y-%m-%d %H:%M:%S %z')
                dates.append(date)
                values.append(value)
            except (ValueError, TypeError):
                continue
    
    print(f"Found {len(dates)} records")
    return pd.DataFrame({'date': dates, 'value': values})

def analyze_steps():
    """
    Analyze and visualize daily step count data.
    Shows a time series plot of daily total steps and exports data to CSV.
    """
    record_type = 'HKQuantityTypeIdentifierStepCount'  # Define record_type
    df = parse_health_data('export.xml', record_type)
    
    # Check if any step data was found
    if len(df) == 0:
        print("No step data found in the export file.")
        # Create an empty CSV file to indicate processing was attempted
        pd.DataFrame(columns=['date', 'value']).to_csv('steps_data.csv', index=False)
        print("Created empty steps_data.csv file.")
        return
    
    # Daily sum of steps
    daily_steps = df.groupby(df['date'].dt.date)['value'].sum()
    
    # Export to CSV
    daily_steps.to_csv('steps_data.csv', header=True)
    print("Steps data exported to 'steps_data.csv'")
    
    # Plot
    plt.figure(figsize=(12, 6))
    daily_steps.plot()
    plt.title('Daily Steps')
    plt.xlabel('Date')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.show()

def analyze_distance():
    """
    Analyze and visualize daily walking/running distance.
    Shows a time series plot of daily total distance in kilometers and exports data to CSV.
    """
    df = parse_health_data('export.xml', 'HKQuantityTypeIdentifierDistanceWalkingRunning')
    
    # Check if any distance data was found
    if len(df) == 0:
        print("No distance data found in the export file.")
        # Create an empty CSV file to indicate processing was attempted
        pd.DataFrame(columns=['date', 'value']).to_csv('distance_data.csv', index=False)
        print("Created empty distance_data.csv file.")
        return
    
    # Daily sum of distance (in kilometers)
    daily_distance = df.groupby(df['date'].dt.date)['value'].sum() / 1000
    
    # Export to CSV
    daily_distance.to_csv('distance_data.csv', header=True)
    print("Distance data exported to 'distance_data.csv'")
    
    # Plot
    plt.figure(figsize=(12, 6))
    daily_distance.plot()
    plt.title('Daily Walking/Running Distance')
    plt.xlabel('Date')
    plt.ylabel('Distance (km)')
    plt.grid(True)
    plt.show()

def analyze_heart_rate():
    """
    Analyze and visualize daily heart rate data with interactive Plotly charts.
    Shows a time series plot of daily average heart rate in BPM, applies rolling averages,
    and exports data to CSV.
    """
    df = parse_health_data('export.xml', 'HKQuantityTypeIdentifierHeartRate')
    
    # Check if any heart rate data was found
    if len(df) == 0:
        print("No heart rate data found in the export file.")
        # Create an empty CSV file to indicate processing was attempted
        pd.DataFrame(columns=['date', 'value']).to_csv('heart_rate_data.csv', index=False)
        print("Created empty heart_rate_data.csv file.")
        return
    
    # Daily average heart rate
    daily_hr = df.groupby(df['date'].dt.date)['value'].mean()
    
    # Apply rolling average for smoother trends (e.g., 7-day rolling average)
    rolling_avg = daily_hr.rolling(window=7).mean()
    
    # Export to CSV
    daily_hr.to_csv('heart_rate_data.csv', header=True)
    print("Heart rate data exported to 'heart_rate_data.csv'")
    
    # Plot using Plotly
    fig = px.line(
        x=daily_hr.index, 
        y=daily_hr.values, 
        title='Daily Average Heart Rate',
        labels={'x': 'Date', 'y': 'Heart Rate (BPM)'},
        color_discrete_sequence=['blue']  # Default color
    )
    
    # Add rolling average to the plot
    fig.add_scatter(
        x=rolling_avg.index, 
        y=rolling_avg.values, 
        mode='lines', 
        name='7-Day Rolling Average',
        line=dict(color='orange', dash='dash')  # Custom color and style
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Heart Rate (BPM)',
        template='plotly_white',
        legend_title='Legend'
    )
    
    # Show the interactive chart
    fig.show()
    """
    Analyze health data using a local Ollama LLM.
    
    Args:
        csv_files: List of CSV files to analyze
    """
    try:
        # Check if required data files exist and run analyses if needed
        missing_files = []
        for file_name, data_type in csv_files:
            if not os.path.exists(file_name):
                missing_files.append((file_name, data_type))
        
        if missing_files:
            print("\nSome required data files are missing. Running analyses to generate them...")
            print("Note: This will generate all required data files without displaying plots.")
            print("You can view the plots later by running options 1-6 individually.")
            
            # Temporarily disable plot display to avoid blocking
            original_show = plt.show
            plt.show = lambda: None  # Replace with no-op function
            
            try:
                # Map file names to their corresponding analysis functions
                analysis_functions = {
                    'steps_data.csv': analyze_steps,
                    'distance_data.csv': analyze_distance,
                    'heart_rate_data.csv': analyze_heart_rate
                }
                
                # Run the necessary analyses
                for file_name, data_type in missing_files:
                    if file_name in analysis_functions:
                        print(f"\nGenerating {file_name} from {data_type} data...")
                        analysis_functions[file_name]()
                        # Verify the file was created
                        if os.path.exists(file_name):
                            print(f"✓ Successfully generated {file_name}")
                        else:
                            print(f"✗ Failed to generate {file_name}")
            finally:
                # Restore original plt.show function
                plt.show = original_show
        
        # Add data preparation code
        data_summary = {}
        files_found = False
        
        print("\nProcessing data files...")
        for file_name, data_type in csv_files:
            try:
                if os.path.exists(file_name):
                    df = pd.read_csv(file_name)
                    
                    # Skip empty dataframes
                    if len(df) == 0:
                        print(f"Note: {file_name} exists but contains no data.")
                        continue
                    
                    print(f"Found {data_type} data in {file_name}")
                    
                    data_summary[data_type] = {
                        'total_records': len(df),
                        'date_range': f"from {df['date'].min()} to {df['date'].max()}" if 'date' in df and len(df) > 0 else 'N/A',
                        'average': f"{df['value'].mean():.2f}" if 'value' in df and len(df) > 0 else 'N/A',
                        'max_value': f"{df['value'].max():.2f}" if 'value' in df and len(df) > 0 else 'N/A',
                        'min_value': f"{df['value'].min():.2f}" if 'value' in df and len(df) > 0 else 'N/A',
                        'data_sample': df.head(50).to_string() if len(df) > 0 else 'No data available'
                    }
                    files_found = True
                else:
                    print(f"Warning: {file_name} still not found after attempted generation.")
                    
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

        if not files_found:
            print("\nNo data files with content could be processed! Please check your export.xml file.")
            print("It appears your Apple Health export doesn't contain the expected health metrics.")
            return

        # Build the prompt
        prompt = "Analyze this Apple Health data and provide detailed insights:\n\n"
        for data_type, summary in data_summary.items():
            prompt += f"\n{data_type} Data Summary:\n"
            prompt += f"- Total Records: {summary['total_records']}\n"
            prompt += f"- Date Range: {summary['date_range']}\n"
            prompt += f"- Average Value: {summary['average']}\n"
            prompt += f"- Maximum Value: {summary['max_value']}\n"
            prompt += f"- Minimum Value: {summary['min_value']}\n"
            prompt += f"\nSample Data:\n{summary['data_sample']}\n"
            prompt += "\n" + "="*50 + "\n"

        prompt += """Please provide a comprehensive analysis including:
        1. Notable patterns or trends in the data
        2. Unusual findings or correlations between different metrics
        3. Actionable health insights based on the data
        4. Areas that might need attention or improvement
        """

        # Rest of the Ollama API call
        print("\nSending data to Deepseek-R1 via Ollama...")
        response = ollama.chat(
            model='deepseek-r1',
            messages=[{
                "role": "system",
                "content": "You are a health data analyst with strong technical skills. Provide detailed analysis with a focus on data patterns, statistical insights, and code-friendly recommendations. Use markdown formatting for technical terms."
            }, {
                "role": "user", 
                "content": prompt
            }],
            options={
                'temperature': 0.3,
                'num_ctx': 6144
            }
        )

        analysis_content = response['message']['content']
        
        print("\nDeepseek-R1 Analysis:")
        print("=" * 50)
        print(analysis_content)
        
        # Ask if user wants to save the analysis
        save_option = input("\nWould you like to save this analysis as a markdown file? (y/n): ").strip().lower()
        if save_option == 'y' or save_option == 'yes':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_analysis_ollama_{timestamp}.md"
            
            # Create markdown content
            markdown_content = f"# Apple Health Data Analysis (Ollama Deepseek-R1)\n\n"
            markdown_content += f"*Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            markdown_content += f"## Data Summary\n\n"
            
            for data_type, summary in data_summary.items():
                markdown_content += f"### {data_type}\n\n"
                markdown_content += f"- **Total Records:** {summary['total_records']}\n"
                markdown_content += f"- **Date Range:** {summary['date_range']}\n"
                markdown_content += f"- **Average Value:** {summary['average']}\n"
                markdown_content += f"- **Maximum Value:** {summary['max_value']}\n"
                markdown_content += f"- **Minimum Value:** {summary['min_value']}\n\n"
            
            markdown_content += f"## Analysis Results\n\n"
            markdown_content += analysis_content
            
            # Save to file
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, 'w') as f:
                f.write(markdown_content)
            
            print(f"\nAnalysis saved to {filepath}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")

def main():
    """
    Main function providing an interactive menu to choose which health metric to analyze.
    """
    while True:
        print("\nWhat would you like to analyze?")
        print("1. Steps")
        print("2. Distance")
        print("3. Heart Rate")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        # List of available data files and their types
        data_files = [
            ('steps_data.csv', 'Steps'),
            ('distance_data.csv', 'Distance'),
            ('heart_rate_data.csv', 'Heart Rate')
        ]
        
        if choice == '1':
            analyze_steps()
        elif choice == '2':
            analyze_distance()
        elif choice == '3':
            analyze_heart_rate()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import pandas
        import matplotlib
        import openai
        from dotenv import load_dotenv
        print("All required packages are installed!")
    except ImportError as e:
        print(f"Missing required package: {str(e)}")
        print("\nPlease install required packages using:")
        print("pip install -r ../requirements.txt")
        exit(1)

def check_env():
    """Check if .env file exists and contains API key"""
    if not os.path.exists('.env'):
        print("Warning: .env file not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your-api-key-here")
        return False
    return True

if __name__ == "__main__":
    check_requirements()
    if not check_env():
        print("\nContinuing without AI analysis capabilities...")
    main()