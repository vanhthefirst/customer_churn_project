"""
Setup script for Customer Churn Prediction project.
This script ensures the correct directory structure and trains the model.
"""
import os
import sys
import subprocess
import shutil

def create_directory_structure():
    """Create the necessary directory structure if it doesn't exist"""
    print("Setting up directory structure...")
    
    # Get current directory (where this script is run from)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create required directories
    required_dirs = ['data', 'models', 'src', 'app']
    for directory in required_dirs:
        dir_path = os.path.join(current_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    return current_dir

def verify_data_file(project_dir):
    """Verify that the data file exists"""
    print("Checking for data file...")
    
    # Define potential filenames
    possible_filenames = [
        'WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'WA_FnUseC_TelcoCustomerChurn.csv'
    ]
    
    # Check for data file in the data directory
    data_dir = os.path.join(project_dir, 'data')
    data_file_path = None
    
    for filename in possible_filenames:
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            data_file_path = file_path
            print(f"Found data file: {data_file_path}")
            break
    
    if not data_file_path:
        print("Error: Data file not found in the data directory!")
        
        # Check if the file exists in the current directory
        for filename in possible_filenames:
            file_path = os.path.join(project_dir, filename)
            if os.path.isfile(file_path):
                # Move the file to the data directory
                destination = os.path.join(data_dir, filename)
                shutil.move(file_path, destination)
                print(f"Moved data file from {file_path} to {destination}")
                data_file_path = destination
                break
    
    if not data_file_path:
        print("Please download the Telco Customer Churn dataset and place it in the 'data' directory.")
        print("You can download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return False
    
    return True

def train_model(project_dir):
    """Train the churn prediction model"""
    print("Training the model...")
    
    # Get the path to run.py
    run_script = os.path.join(project_dir, 'run.py')
    
    if not os.path.isfile(run_script):
        print(f"Error: run.py not found at {run_script}")
        return False
    
    try:
        # Execute the training command
        subprocess.run([sys.executable, run_script, '--train'], check=True)
        print("Model training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during model training: {e}")
        return False

def main():
    """Main function to set up the project and train the model"""
    print("=" * 50)
    print("Customer Churn Prediction & Retention Analysis Setup")
    print("=" * 50)
    
    # Create directory structure
    project_dir = create_directory_structure()
    
    # Verify data file
    if not verify_data_file(project_dir):
        return
    
    # Train the model
    if train_model(project_dir):
        print("\nSetup completed successfully!")
        print("\nYou can now run the dashboard using:")
        print(f"python {os.path.join(project_dir, 'run.py')} --dashboard")
    else:
        print("\nSetup encountered errors. Please check the messages above.")

if __name__ == "__main__":
    main()